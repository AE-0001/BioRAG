from biorag.paddle_ocr import _find_recognized_text


def test_extracts_paddle_v3_recognition_text():
    payload = {
        "res": {
            "overall_ocr_res": {
                "rec_texts": ["B7-H3 expression", "response rate 61%"],
                "rec_scores": [0.99, 0.95],
            }
        }
    }
    assert _find_recognized_text(payload) == ["B7-H3 expression", "response rate 61%"]

