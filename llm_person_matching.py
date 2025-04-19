#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Literal

from ollama import Client
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic モデル定義: 2 画像間の同一人物判定結果のスキーマ
# ─────────────────────────────────────────────────────────────────────────────
class TwoPersonSimilarity(BaseModel):
    # 同一人物かどうかの判定: 'yes' または 'no'
    same_person: Literal["yes", "no"]
    # 判定の信頼度 (0～100)
    similarity_confidence: float = Field(ge=0, le=100)


# ─────────────────────────────────────────────────────────────────────────────
# メイン処理: 2 枚の画像パスを受け取り、LLM に問い合わせて判定結果を取得・検証・表示
# ─────────────────────────────────────────────────────────────────────────────
def main(image1_path: str, image2_path: str):
    """
    Args:
        image1_path: 比較対象となる最初の画像ファイルへのパス
        image2_path: 比較対象となる二番目の画像ファイルへのパス

    処理概要:
      1. 両方のファイル存在チェック
      2. Ollama クライアントの初期化
      3. JSON スキーマの生成
      4. LLM に問い合わせ、2 画像の同一人物判定を実施
      5. 結果を Pydantic で検証・整形し JSON 表示
    """
    # 1. 画像ファイルの存在検証
    img1 = Path(image1_path)
    img2 = Path(image2_path)
    if not img1.exists() or not img2.exists():
        missing = []
        if not img1.exists():
            missing.append(image1_path)
        if not img2.exists():
            missing.append(image2_path)
        sys.exit(f"Error: file not found: {', '.join(missing)}")

    # 2. Ollama サーバーへの接続クライアントを作成
    client = Client(host="http://10.0.1.5:11434")

    # 3. Pydantic モデルから JSON Schema を自動生成
    schema = TwoPersonSimilarity.model_json_schema()

    # 4. LLM に問い合わせ (chat API 呼び出し)
    response = client.chat(
        model="gemma3:4b",  # 使用モデル名
        messages=[
            {
                "role": "system",
                "content": "Respond only with JSON matching the schema.",
            },
            {
                "role": "user",
                "content": (
                    "Given two images, determine if they depict the same person. "
                    "Answer 'yes' or 'no' in the 'same_person' field, "
                    "and provide a confidence score (0-100) in 'similarity_confidence'."
                ),
                # 2 画像をアップロードするためのフィールド
                "images": [image1_path, image2_path],
            },
        ],
        format=schema,  # レスポンス形式を JSON Schema に合わせる
        options={"temperature": 0},  # 再現性のため温度パラメータを 0 に設定
    )

    # 5. 受信したレスポンスを Pydantic で検証・パース
    result = TwoPersonSimilarity.model_validate_json(response.message.content)

    # 整形した結果を標準出力に JSON で表示
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


# エントリポイント: コマンドライン引数をチェックして main() を実行
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} <image1_path> <image2_path>")
    main(sys.argv[1], sys.argv[2])
