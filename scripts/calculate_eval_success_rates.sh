#!/bin/bash

set -euo pipefail

SUMMARY_FILE="$1"

if [ ! -f "$SUMMARY_FILE" ]; then
    echo "Error: Summary file not found at $SUMMARY_FILE"
    exit 1
fi

echo "📊 Evaluation Success Rate Analysis"
echo "================================================================================"
echo ""

# Evaluation result
awk '
/## 📝 Evaluation Results/ {
    in_results_section = 1
    next
}
in_results_section && /Resolution Rate/ {
    rate_str = $0
    sub(/.*: /, "", rate_str)
    print "### Evaluation Resolution Rate: " rate_str
    in_results_section = 0 # Stop after finding it
}' "$SUMMARY_FILE"

echo ""

# Overall tool success rate
awk '
/Overall Calls/ {
    line = $0;
    sub(/.*Overall Calls.*: /, "", line);
    
    total_calls_str = line;
    sub(/ .*/, "", total_calls_str);
    
    successful_calls_str = line;
    sub(/.*Successful: /, "", successful_calls_str);
    sub(/,.*/, "", successful_calls_str);

    total_calls = total_calls_str + 0;
    successful_calls = successful_calls_str + 0;

    if (total_calls > 0) {
        rate = (successful_calls / total_calls) * 100;
        printf("### Overall Tool Success Rate: %.2f%% (%d/%d)\n", rate, successful_calls, total_calls);
    } else {
        print "### Overall Tool Success Rate: N/A (0 calls)";
    }
}' "$SUMMARY_FILE"

echo ""
echo "### Tool-specific Success Rates"
echo "--------------------------------"

# Tool-specific success rates
awk '
/  - **Tool: `.*`**/ {
    line = $0;
    start = index(line, "`") + 1;
    end = index(substr(line, start), "`");
    tool_name = substr(line, start, end - 1);

    getline; calls_line = $0;
    sub(/.*Calls: /, "", calls_line);
    calls = calls_line + 0;

    getline; successful_line = $0;
    sub(/.*Successful: /, "", successful_line);
    successful = successful_line + 0;

    if (calls > 0) {
        rate = (successful / calls) * 100;
        printf("- **%s**: %.2f%% (%d/%d)\n", tool_name, rate, successful, calls);
    } else {
        printf("- **%s**: N/A (0 calls)\n", tool_name);
    }
}
' "$SUMMARY_FILE"


# Localization Metrics
echo ""
echo "### Localization Metrics"
echo "--------------------------------"

SOLUTION_DIR=$(dirname "$(dirname "$SUMMARY_FILE")")

# Run embedded python script for localization metrics
python - "$SOLUTION_DIR" <<EOF
import json
import sys
from pathlib import Path

# These dependencies are expected to be in the environment where the script runs.
from agent_prototypes.patching import extract_filenames_from_patch
from agent_prototypes.swebench.swebench_instance import SWEBenchInstance
from agent_prototypes.utils.utils import REPO_ROOT

DATASET_REL_PATH = "data/swebench-lite/test-00000-of-00001.parquet"

def evaluate_localization(solution_path: str, dataset_rel_path = DATASET_REL_PATH) -> tuple[float, float]:
    output_file = Path(solution_path) / "output.jsonl"
    if not output_file.is_file():
        # Don't print warnings if the file just doesn't exist for this run
        return 0.0, 0.0

    with open(output_file) as fin:
        solutions = [json.loads(line) for line in fin.readlines()]

    if not solutions:
        return 0.0, 0.0

    dataset_path = str(REPO_ROOT / dataset_rel_path)
    data = SWEBenchInstance.from_parquet(Path(dataset_path))

    instance_recalls = []
    instance_precs = []
    for solution in solutions:
        predicted_loc = extract_filenames_from_patch(solution.get("model_patch", ""))
        instance_list = [d for d in data if d.instance_id == solution["instance_id"]]
        
        if not instance_list:
            continue
        instance = instance_list[0]

        gt_loc = extract_filenames_from_patch(instance.patch)
        
        recall = 1.0
        if len(gt_loc) > 0:
            recall = len(set(predicted_loc) & set(gt_loc)) / len(gt_loc)
        elif len(predicted_loc) > 0:
            recall = 0.0
        instance_recalls.append(recall)

        prec = 1.0
        if len(predicted_loc) > 0:
            prec = len(set(predicted_loc) & set(gt_loc)) / len(predicted_loc)
        instance_precs.append(prec)

    avg_recall = sum(instance_recalls) / len(instance_recalls) if instance_recalls else 0.0
    avg_prec = sum(instance_precs) / len(instance_precs) if instance_precs else 0.0
    
    return avg_recall, avg_prec

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Exit gracefully if no dir is provided
        sys.exit(0)
    
    solution_dir_arg = sys.argv[1]
    recall, prec = evaluate_localization(solution_dir_arg)
    
    # Only print if there were results
    if recall > 0 or prec > 0:
        print(f"- **Average Recall**: {recall:.2f}")
        print(f"- **Average Precision**: {prec:.2f}")
EOF
