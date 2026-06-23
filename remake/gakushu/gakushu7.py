import pandas as pd
import lightgbm as lgb
import joblib
import os

# --- 1. 学習パート ---
def train_model():
    print("--- モデル学習開始 ---")
    df = pd.read_csv('model_data.csv')
    
    # ターゲット列の定義
    target_cols = ['is_1st', 'is_top2', 'is_top3']
    
    # 特徴量から学習に使わないものを除外
    # (単勝・人気はリーケージ防止のため学習にも使用しない)
    exclude = ['is_1st', 'is_top2', 'is_top3', 'レースID', 'レース名', '馬名', '馬主', 'タイム', '着差', '通過順', '賞金', '性別年齢', '上り3ハロン', '単勝', '人気', '着順', '上がり偏差値']
    feature_cols = [c for c in df.columns if c not in exclude]

    for t_col in target_cols:
        model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'verbose': -1},
            lgb.Dataset(df[feature_cols].astype(float), df[t_col]),
            num_boost_round=500
        )
        model.save_model(f'model_{t_col}.txt')
        print(f"モデル '{t_col}' 保存完了")

# --- 2. 予測パート ---
def predict_today():
    print("\n--- 本日のレース予測開始 ---")
    today_df = pd.read_csv('today_race.csv') # 名前が入っている前提
    
    # 保存した変換ルール(pkl)を使って名前を数値に変換
    le_cols = ['天気', '馬場状態', 'レース条件', '芝ダート', '回り', '騎手', '調教師名', '所属', '性別', '曜日']
    for col in le_cols:
        if os.path.exists(f'le_{col}.pkl'):
            le = joblib.load(f'le_{col}.pkl')
            # 予測データ内の文字列をID変換（未学習の名前は0）
            today_df[col] = today_df[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

    # 推論用の特徴量選択
    feature_cols = [c for c in today_df.columns if c not in ['レース名', '馬名', 'レース目', 'レースID']]
    
    # 予測実行
    for t_col in ['is_1st', 'is_top2', 'is_top3']:
        booster = lgb.Booster(model_file=f'model_{t_col}.txt')
        today_df[t_col] = booster.predict(today_df[feature_cols].astype(float))

    # 結果表示
    output = today_df[['レース目', '馬名', 'is_1st', 'is_top2', 'is_top3']].sort_values(['レース目', 'is_1st'], ascending=[True, False])
    print(output.to_string(index=False))
    output.to_csv('prediction_result.csv', index=False, encoding='utf_8_sig')
    print("\n予測結果を prediction_result.csv に保存しました。")

if __name__ == "__main__":
    # 学習済みモデルがあれば予測へ、なければ学習してから予測へ
    if os.path.exists('model_is_1st.txt'):
        predict_today()
    else:
        train_model()
        predict_today()
    
    input("\n処理が完了しました。Enterキーを押して終了してください...")