LABEL_TOOL_DIR="C:\Users\youjc04\Documents\WXWork\1688857891796034\WeDrive\万科\中广核标记结果\OCR标记"
DATASET_NAME="A"
CROPPED_DIR="E:\datasets\CGN\test\rec_test_$DATASET_NAME"
# MODEL_PATH="E:\models\PP-OCRv5_server_rec_infer"
MODEL_PATH="E:\models\inference"
OUTPUT_DIR="E:\datasets\CGN\test\rec_test_$DATASET_NAME\\$DATASET_NAME_result"


function crop_images() {
    python scripts/crop_images.py \
        --label_tool_dir "$LABEL_TOOL_DIR" \
        --output_dir "$CROPPED_DIR" \
        --dataset_name "$DATASET_NAME"
}
function run_pp_rec() {
    python tests/test_pp_rec.py \
        --model_path "$MODEL_PATH" \
        --test_dir "$CROPPED_DIR/Images" \
        --output_json_path "$OUTPUT_DIR/results.json"
}

function run_rec_score() {
    python rec_score.py \
        --pred_path "$OUTPUT_DIR/results.json" \
        --label_path "$CROPPED_DIR/Label.json" \
        --output_path "$OUTPUT_DIR/results.csv"
}

function vis_badcaes(){
    python scripts/export_badcase.py \
    --label_path "$CROPPED_DIR/Label.json" \
    --pred_path "$OUTPUT_DIR/results.json" \
    --image_dir "$CROPPED_DIR/Images" \
    --output "$OUTPUT_DIR/badcases.xlsx" \
    --format xlsx \
    --only_diff
}


crop_images
run_pp_rec
run_rec_score
vis_badcaes