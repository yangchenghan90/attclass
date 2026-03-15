import pandas as pd
import re

# 读取数据
df = pd.read_excel('正确组.xlsx')
df.columns = ['推荐选项列表', '实际选项', '选项类型']

# 文本清洗函数
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
# 应用清洗
df['清理后实际选项'] = df['实际选项'].apply(clean_text)
df['清理后推荐选项'] = df['推荐选项列表'].apply(clean_text)