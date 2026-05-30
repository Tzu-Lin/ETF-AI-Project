import joblib

models_to_check = ["models/rf_SPY.joblib", "models/rf_QQQ.joblib", "models/rf_0050_tw.joblib"]
for path in models_to_check:
    try:
        obj = joblib.load(path)
        print(f"\n档案：{path}")
        print(f"  型态：{type(obj)}")
        if isinstance(obj, dict):
            print("  是字典，键有：", list(obj.keys()))
            if 'model' in obj:
                print("  包含 'model' 键，其型态为：", type(obj['model']))
                if hasattr(obj['model'], 'predict_proba'):
                    print("  ✅ 可以透过 obj['model'] 取得分类器")
            if 'estimator' in obj:
                print("  包含 'estimator' 键，其型态为：", type(obj['estimator']))
        elif hasattr(obj, 'predict_proba'):
            print("  ✅ 直接是分类器，可用")
        else:
            print("  ❌ 无法识别的格式")
    except Exception as e:
        print(f"读取 {path} 失败：{e}")