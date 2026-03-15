import pandas as pd
import numpy as np
import re
import json
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences





# ======================== 配置 ========================
MAX_SEQ_LEN = 10

# ======================== 加载模型和工具 ========================
# 加载模型
model = load_model('beverage_recommendation_model.h5')

# 加载tokenizer
with open('tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)

# 加载编码器
type_encoder = joblib.load('type_encoder.pkl')
target_encoder = joblib.load('target_encoder.pkl')


# ======================== 工具函数 ========================
def clean_text(text):
    """增强版文本清洗函数 - 完整覆盖甜度表述"""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # 去除特殊字符和标点（保留中文、数字、空格）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)

    # ========== 完整的甜度统一规则 ==========
    # 1. 核心甜度数值统一（数字+分糖）
    text = re.sub(r'五分糖|五分|5分|五份糖|五糖', '5分糖', text)
    text = re.sub(r'三分糖|三分|3分|三份糖|三糖', '3分糖', text)
    text = re.sub(r'七分糖|七分|7分|七份糖|七糖', '7分糖', text)
    text = re.sub(r'九分糖|九分|9分|九份糖|九糖', '9分糖', text)
    text = re.sub(r'一分糖|一分|1分|一份糖|一糖', '1分糖', text)

    # 2. 全糖/正常糖统一
    text = re.sub(r'全糖|正常糖|标准糖|十分糖|满分糖|10分糖|十成糖', '全糖', text)

    # 3. 无糖/零糖统一
    text = re.sub(r'无糖|零糖|0糖|不加糖|不另外加糖|无额外糖', '无糖', text)

    # 4. 半糖/中糖统一
    text = re.sub(r'半糖|中糖|5成糖|五成糖', '半糖', text)

    # 5. 微糖/少糖统一
    text = re.sub(r'微糖|少糖|微甜|少量糖|轻糖', '微糖', text)

    # 6. 多糖/加甜统一
    text = re.sub(r'多糖|加甜|超甜|多糖份|额外糖', '多糖', text)

    # ========== 温度表述统一（补充） ==========
    # 避免只处理甜度，遗漏温度的统一
    text = re.sub(r'正常冰|标准冰|常规冰', '正常冰', text)
    text = re.sub(r'少冰|减冰|半冰', '少冰', text)
    text = re.sub(r'去冰|无冰|不加冰', '去冰', text)
    text = re.sub(r'热饮|热的|温的|常温', '热饮', text)

    return text.strip()


def preprocess_input(recommendation_text, option_type):
    """预处理输入数据"""
    # 文本清洗
    cleaned_text = clean_text(recommendation_text)

    # 文本序列转换
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    X_seq = pad_sequences(sequence, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

    # 类型特征编码
    type_df = pd.DataFrame([[option_type]], columns=['选项类型'])
    try:
        X_type = type_encoder.transform(type_df)
    except:
        # 如果类型不在编码器中，返回全0
        X_type = np.zeros((1, type_encoder.categories_[0].shape[0] - 1))

    # 合并特征
    X = np.hstack((X_seq, X_type))

    return X


def predict_beverage_option(recommendation_text, option_type):
    """
    预测饮料选项

    参数:
        recommendation_text: 推荐选项文本
        option_type: 选项类型（如"甜度"、"温度"、"奶茶底"）

    返回:
        dict: 预测结果
    """
    # 预处理
    X = preprocess_input(recommendation_text, option_type)

    # 预测
    pred_proba = model.predict(X, verbose=0)[0]
    pred_index = np.argmax(pred_proba)
    pred_option = target_encoder.inverse_transform([pred_index])[0]

    # 生成详细结果
    result = {
        '输入推荐文本': recommendation_text,
        '选项类型': option_type,
        '预测选项': pred_option,
        '置信度': round(float(pred_proba[pred_index]), 4),
        '是否可信': float(pred_proba[pred_index]) >= 0.5,
        '所有选项置信度': {}
    }

    # 所有选项的置信度
    for i, cls in enumerate(target_encoder.classes_):
        result['所有选项置信度'][cls] = round(float(pred_proba[i]), 4)

    return result


# ======================== 命令行交互 ========================
def interactive_predict():
    """交互式预测"""
    print("=" * 60)
    print("🎯 饮料推荐选项预测工具")
    print("=" * 60)
    print("支持的选项类型: 甜度、温度、奶茶底、温馨提示等")
    print("输入 'quit' 或 'exit' 退出程序")
    print("-" * 60)

    while True:
        # 获取用户输入
        rec_text = input("\n请输入推荐选项文本: ")
        if rec_text.lower() in ['quit', 'exit', '退出']:
            print("👋 程序结束")
            break

        option_type = input("请输入选项类型: ")
        if option_type.lower() in ['quit', 'exit', '退出']:
            print("👋 程序结束")
            break

        # 预测
        try:
            result = predict_beverage_option(rec_text, option_type)

            # 显示结果
            print("\n" + "=" * 40)
            print("📊 预测结果")
            print("=" * 40)
            print(f"预测选项: {result['预测选项']}")
            print(f"置信度: {result['置信度']} ({'✅ 可信' if result['是否可信'] else '⚠️  低置信度'})")
            print(f"选项类型: {result['选项类型']}")

            # 显示前3个高置信度选项
            print("\n📈 置信度排名前3:")
            sorted_probs = sorted(
                result['所有选项置信度'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            for i, (opt, prob) in enumerate(sorted_probs, 1):
                print(f"  {i}. {opt}: {prob}")

        except Exception as e:
            print(f"\n❌ 预测出错: {str(e)}")


# ======================== 批量预测函数 ========================
def batch_predict_from_excel(file_path, output_file='预测结果.xlsx'):
    """
    从Excel文件批量预测

    参数:
        file_path: 输入Excel文件路径
        output_file: 输出文件路径
    """
    # 读取数据
    df = pd.read_excel(file_path)
    df.columns = ['推荐选项列表', '实际选项', '选项类型']

    # 批量预测
    predictions = []
    confidences = []

    for _, row in df.iterrows():
        try:
            result = predict_beverage_option(row['推荐选项列表'], row['选项类型'])
            predictions.append(result['预测选项'])
            confidences.append(result['置信度'])
        except:
            predictions.append('预测失败')
            confidences.append(0.0)

    # 添加预测结果
    df['预测选项'] = predictions
    df['预测置信度'] = confidences
    df['预测是否可信'] = df['预测置信度'] >= 0.5

    # 保存结果
    df.to_excel(output_file, index=False)
    print(f"✅ 批量预测完成，结果已保存至: {output_file}")

    # 统计结果
    success_rate = (df['预测是否可信'].sum() / len(df)) * 100
    print(f"📊 预测统计:")
    print(f"   - 总预测数: {len(df)}")
    print(f"   - 高置信度预测: {df['预测是否可信'].sum()}")
    print(f"   - 高置信度率: {success_rate:.1f}%")


# ======================== 主函数 ========================
if __name__ == "__main__":
    # 交互式预测
    print("111")
    interactive_predict()

    # 如需批量预测，取消注释以下代码
    # batch_predict_from_excel('待预测数据.xlsx')