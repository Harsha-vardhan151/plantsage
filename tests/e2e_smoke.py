import os
def test_export_exists():
    assert os.path.exists("server/models/classifier.onnx") or os.path.exists("server/models/classifier.ts.pt")