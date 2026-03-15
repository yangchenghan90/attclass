import pandas as pd
import numpy as np
import re
import json
import random
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Dense, Dropout, Embedding, Flatten, Input,
                                     Concatenate, BatchNormalization, LSTM)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# ======================== 配置参数（优化后） ========================
CONFIG = {
    'EXCEL_FILE': '正确组.xlsx',
    'MAX_SEQ_LEN': 20,  # 增加序列长度，保留更多文本信息
    'VOCAB_SIZE': 2000,  # 扩大词汇表
    'EPOCHS': 80,  # 增加训练轮数（配合早停）
    'BATCH_SIZE': 16,  # 调大批次，提升稳定性
    'TEST_SIZE': 0.2,  # 合理提升测试集比例
    'VALIDATION_SPLIT': 0.2,
    'AUGMENT_FACTOR': 6,  # 增加增强倍数
    'RANDOM_STATE': 42
}

# 设置中文字体（解决图表中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ======================== 工具函数（深度优化） ========================
def clean_text(text):
    """增强版文本清洗：保留语义+统一表述+去重字符"""
    if pd.isna(text):
        return ""
    text = str(text).strip()

    # 仅去除极端特殊符号，保留核心语义字符
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。、]', '', text)

    # ========== 甜度表述统一（新增模糊匹配） ==========
    text = re.sub(r'五分糖|五分|5分|五份糖|五糖', '5分糖', text)
    text = re.sub(r'三分糖|三分|3分|三份糖|三糖', '3分糖', text)
    text = re.sub(r'七分糖|七分|7分|七份糖|七糖', '7分糖', text)
    text = re.sub(r'九分糖|九分|9分|九份糖|九糖', '9分糖', text)
    text = re.sub(r'一分糖|一分|1分|一份糖|一糖', '1分糖', text)
    text = re.sub(r'全糖|正常糖|标准糖|十分糖|满分糖|10分糖|十成糖', '全糖', text)
    text = re.sub(r'无糖|零糖|0糖|不加糖|不另外加糖|无额外糖', '无糖', text)
    text = re.sub(r'半糖|中糖|5成糖|五成糖', '半糖', text)
    text = re.sub(r'微糖|少糖|微甜|少量糖|轻糖|少少糖|一点点糖', '微糖', text)
    text = re.sub(r'多糖|加甜|超甜|多糖份|额外糖|多一点糖', '多糖', text)

    # ========== 温度/冰度表述统一 ==========
    text = re.sub(r'正常冰|标准冰|常规冰', '正常冰', text)
    text = re.sub(r'少冰|减冰|半冰', '少冰', text)
    text = re.sub(r'去冰|无冰|不加冰', '去冰', text)
    text = re.sub(r'热饮|热的|温的|常温', '热饮', text)

    # 去除重复字符（如"多多多冰"→"多冰"）
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    return text.strip()


def merge_categories(option):
    """优化类别合并：减少细分类别，降低学习难度"""
    merge_rules = {
        # 甜度大类合并（核心优化：减少类别数）
        '甜度-中等': ['5分糖', '半糖', '中糖'],
        '甜度-低': ['3分糖', '微糖', '1分糖'],
        '甜度-高': ['7分糖', '9分糖', '全糖', '多糖'],
        '甜度-无': ['无糖'],
        # 冰度大类合并
        '冰度-正常': ['正常冰', '正常冰推荐', '手捣正常冰'],
        '冰度-少': ['少冰', '少冰推荐'],
        '冰度-无': ['去冰'],
        '温度-热': ['热饮'],
        # 基底合并
        '基底-奶茶': ['奶茶底', '经典底', '原味底'],
        '无添加': ['无', '不添加', '不加']
    }

    for target, sources in merge_rules.items():
        if any(source in option for source in sources):
            return target
    return '其他'


def augment_data(df, augment_factor=6):
    """增强版数据增强：多策略文本扰动+去重"""
    augmented_data = []

    # 扩展替换词表
    replace_words = {
        '推荐': ['建议', '推荐选择', '推荐配置', '推荐口味', '更推荐'],
        '口感': ['口味', '味道', '风味', '口感体验', '喝起来'],
        '淡': ['偏淡', '清淡', '不甜', '甜度低', '微甜'],
        '甜': ['偏甜', '较甜', '甜度高', '甜一点'],
        '冰': ['冰块', '冰度', '冰量'],
        '糖': ['甜度', '糖度', '糖量']
    }

    # 随机插入/删除字符
    def random_insert_delete(text):
        if len(text) < 3:
            return text
        chars = list(text)
        core_words = ['糖', '冰', '分', '全', '无', '半', '少', '热']

        if random.random() < 0.5:
            # 插入：随机位置插入相关词汇
            insert_words = ['适中', '刚好', '推荐', '口感']
            pos = random.randint(0, len(text))
            return text[:pos] + random.choice(insert_words) + text[pos:]
        else:
            # 删除：避免删除核心词汇
            del_candidates = [i for i, c in enumerate(chars) if c not in core_words]
            if del_candidates:
                del_pos = random.choice(del_candidates)
                chars.pop(del_pos)
            return ''.join(chars)

    for _, row in df.iterrows():
        # 保留原始样本
        augmented_data.append(row.to_dict())

        # 生成增强样本
        for _ in range(augment_factor):
            new_row = row.to_dict()
            rec_text = row['清理后推荐选项']

            if rec_text:
                aug_text = rec_text
                # 策略1：词汇替换
                for old, new_words in replace_words.items():
                    if old in aug_text and random.random() < 0.7:
                        aug_text = aug_text.replace(old, random.choice(new_words))
                # 策略2：随机插入/删除
                if random.random() < 0.5:
                    aug_text = random_insert_delete(aug_text)

                new_row['清理后推荐选项'] = aug_text.strip()

            augmented_data.append(new_row)

    # 去重：避免重复样本导致过拟合
    df_aug = pd.DataFrame(augmented_data).drop_duplicates(
        subset=['清理后推荐选项', '合并后实际选项']
    ).reset_index(drop=True)

    return df_aug


# ======================== 数据预处理（新增类别均衡） ========================
def load_and_preprocess_data():
    """加载和预处理数据：新增类别均衡处理"""
    print("=== 1. 加载数据 ===")
    # 读取数据
    df = pd.read_excel(CONFIG['EXCEL_FILE'])
    df.columns = ['推荐选项列表', '实际选项', '选项类型']
    print(f"原始数据量: {len(df)} 条")

    # 文本清洗
    print("\n=== 2. 文本清洗 ===")
    df['清理后实际选项'] = df['实际选项'].apply(clean_text)
    df['清理后推荐选项'] = df['推荐选项列表'].apply(clean_text)

    # 类别合并
    print("\n=== 3. 类别合并 ===")
    df['合并后实际选项'] = df['清理后实际选项'].apply(merge_categories)

    # 过滤无效数据
    df_valid = df[df['合并后实际选项'] != ''].reset_index(drop=True)
    print(f"有效数据量: {len(df_valid)} 条")
    print("合并后类别分布:")
    class_dist = df_valid['合并后实际选项'].value_counts()
    print(class_dist)

    # ========== 新增：类别均衡处理 ==========
    print("\n=== 4. 类别均衡处理 ===")
    # 定义采样策略：少数类过采样，多数类欠采样
    sampling_strategy = {}
    for cls in class_dist.index:
        if class_dist[cls] < 50:
            sampling_strategy[cls] = 50  # 少数类过采样到50条
        elif class_dist[cls] > 200:
            sampling_strategy[cls] = 200  # 多数类欠采样到200条
        else:
            sampling_strategy[cls] = class_dist[cls]

    # 欠采样平衡类别
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy,
                             random_state=CONFIG['RANDOM_STATE'])
    X_res, y_res = rus.fit_resample(
        df_valid[['清理后推荐选项', '选项类型']],
        df_valid['合并后实际选项']
    )
    df_balanced = pd.DataFrame({
        '清理后推荐选项': X_res['清理后推荐选项'],
        '选项类型': X_res['选项类型'],
        '合并后实际选项': y_res
    })
    print(f"均衡后数据量: {len(df_balanced)} 条")
    print("均衡后类别分布：")
    print(df_balanced['合并后实际选项'].value_counts())

    # 数据增强
    print("\n=== 5. 数据增强 ===")
    df_augmented = augment_data(df_balanced, CONFIG['AUGMENT_FACTOR'])
    print(f"增强后数据量: {len(df_augmented)} 条")

    return df_augmented


# ======================== 特征工程（新增TF-IDF特征） ========================
def feature_engineering(df):
    """特征工程：新增TF-IDF特征+动态类别匹配"""
    print("\n=== 6. 特征工程 ===")

    # 6.1 文本序列处理
    tokenizer = Tokenizer(num_words=CONFIG['VOCAB_SIZE'], oov_token='<OOV>')
    tokenizer.fit_on_texts(df['清理后推荐选项'])
    sequences = tokenizer.texts_to_sequences(df['清理后推荐选项'])
    X_seq = pad_sequences(
        sequences,
        maxlen=CONFIG['MAX_SEQ_LEN'],
        padding='post',
        truncating='post'
    )

    # 6.2 新增：TF-IDF特征（捕捉文本语义）
    tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))  # 二元语法
    X_tfidf = tfidf.fit_transform(df['清理后推荐选项']).toarray()

    # 6.3 选项类型编码（独热编码）
    type_encoder = OneHotEncoder(sparse_output=False, drop='first')
    X_type = type_encoder.fit_transform(df[['选项类型']])

    # ========== 动态获取有效类别 ==========
    print("\n=== 7. 处理目标变量 ===")
    df['目标类别'] = df['合并后实际选项'].replace('', '其他')
    unique_classes = df['目标类别'].unique()
    print(f"有效类别数量: {len(unique_classes)}")
    print(f"类别列表: {unique_classes[:10]}...")

    # 标签编码+独热编码
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df['目标类别'])
    y_onehot = to_categorical(y)

    # 验证维度
    print(f"标签编码后维度: {y_onehot.shape}")
    print(f"类别数量 vs 编码维度: {len(unique_classes)} vs {y_onehot.shape[1]}")

    # ========== 合并特征：序列+TF-IDF+类型 ==========
    X = np.hstack((X_seq, X_tfidf, X_type))

    print(f"序列特征维度: {X_seq.shape}")
    print(f"TF-IDF特征维度: {X_tfidf.shape}")
    print(f"类型特征维度: {X_type.shape}")
    print(f"总特征维度: {X.shape}")
    print(f"目标变量维度: {y_onehot.shape}")

    # 保存所有编码器
    with open('tokenizer.json', 'w', encoding='utf-8') as f:
        json.dump(tokenizer.to_json(), f, ensure_ascii=False)
    joblib.dump(type_encoder, 'type_encoder.pkl')
    joblib.dump(target_encoder, 'target_encoder.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

    return X, y_onehot, tokenizer, type_encoder, target_encoder, unique_classes


# ======================== 模型构建与训练（函数式API优化） ========================
def build_and_train_model(X, y):
    """构建和训练模型：函数式API+LSTM+多分支特征"""
    print("\n=== 8. 数据分割 ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG['TEST_SIZE'],
        random_state=CONFIG['RANDOM_STATE'],
        shuffle=True
    )

    # 拆分不同特征分支（关键：分开处理文本/TF-IDF/类型特征）
    seq_len = CONFIG['MAX_SEQ_LEN']
    tfidf_dim = 100  # 对应TF-IDF的max_features
    X_train_seq = X_train[:, :seq_len]  # 文本序列特征
    X_train_tfidf = X_train[:, seq_len:seq_len + tfidf_dim]  # TF-IDF特征
    X_train_type = X_train[:, seq_len + tfidf_dim:]  # 类型特征

    X_test_seq = X_test[:, :seq_len]
    X_test_tfidf = X_test[:, seq_len:seq_len + tfidf_dim]
    X_test_type = X_test[:, seq_len + tfidf_dim:]

    print(f"训练集: {X_train.shape[0]} 条 | 测试集: {X_test.shape[0]} 条")

    print("\n=== 9. 构建模型（函数式API） ===")
    # 分支1：文本序列（LSTM捕捉序列依赖）
    text_input = Input(shape=(seq_len,))
    embedding = Embedding(
        input_dim=CONFIG['VOCAB_SIZE'],
        output_dim=64,  # 提升嵌入维度
        input_length=seq_len,
        embeddings_regularizer=l2(0.001)
    )(text_input)
    lstm_out = LSTM(32, return_sequences=False)(embedding)
    text_branch = Dropout(0.3)(lstm_out)

    # 分支2：TF-IDF特征
    tfidf_input = Input(shape=(tfidf_dim,))
    tfidf_dense = Dense(32, activation='relu')(tfidf_input)
    tfidf_branch = BatchNormalization()(tfidf_dense)

    # 分支3：选项类型特征
    type_input = Input(shape=(X_train_type.shape[1],))
    type_dense = Dense(16, activation='relu')(type_input)
    type_branch = BatchNormalization()(type_dense)

    # 合并所有分支
    concat = Concatenate()([text_branch, tfidf_branch, type_branch])
    dense1 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(concat)
    dense1 = Dropout(0.4)(dense1)
    dense2 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(dense1)
    dense2 = Dropout(0.3)(dense2)
    output = Dense(y.shape[1], activation='softmax')(dense2)

    # 构建模型
    model = Model(
        inputs=[text_input, tfidf_input, type_input],
        outputs=output
    )

    # 优化器（学习率衰减）
    optimizer = Adam(learning_rate=0.001, decay=1e-5)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    print("\n=== 10. 训练模型 ===")
    # 优化回调函数
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',  # 监控验证准确率
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.3,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # 训练模型（传入多分支输入）
    history = model.fit(
        [X_train_seq, X_train_tfidf, X_train_type], y_train,
        epochs=CONFIG['EPOCHS'],
        batch_size=CONFIG['BATCH_SIZE'],
        validation_split=CONFIG['VALIDATION_SPLIT'],
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    return model, history, X_test_seq, X_test_tfidf, X_test_type, y_test


# ======================== 模型评估与保存（修复分类报告） ========================
def evaluate_and_save_model(model, history, X_test_seq, X_test_tfidf, X_test_type,
                            y_test, target_encoder, unique_classes):
    """评估和保存模型：修复类别不匹配+详细评估"""
    print("\n=== 11. 模型评估 ===")
    # 模型预测
    y_pred_proba = model.predict([X_test_seq, X_test_tfidf, X_test_type], verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"测试集准确率: {accuracy:.4f}")

    # 修复分类报告：只显示实际出现的类别
    print("\n分类报告:")
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    class_names = target_encoder.inverse_transform(unique_labels)
    print(classification_report(
        y_true, y_pred,
        labels=unique_labels,
        target_names=class_names,
        zero_division=0
    ))

    # 可视化训练过程
    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失', linewidth=2)
    plt.plot(history.history['val_loss'], label='验证损失', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('损失曲线', fontweight='bold')
    plt.grid(alpha=0.3)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='训练准确率', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
    plt.axhline(y=accuracy, color='red', linestyle='--', label=f'测试准确率: {accuracy:.3f}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('准确率曲线', fontweight='bold')
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 保存模型
    print("\n=== 12. 保存模型 ===")
    model.save('beverage_recommendation_model.h5')
    print("✅ 最优模型已保存为: best_model.h5")
    print("✅ 最终模型已保存为: beverage_recommendation_model.h5")

    # 生成使用说明
    generate_usage_doc(accuracy, unique_classes)

    return accuracy


def generate_usage_doc(accuracy, class_names):
    """生成详细使用说明"""
    # 关键修复：完整闭合三引号，补充完整的markdown内容
    usage_doc = f""" # 饮料推荐模型使用说明
    ...（中间内容）...
   pip install tensorflow pandas scikit-learn numpy matplotlib imbalanced-learn

## 📊 模型信息
- **模型类型**: 多分支LSTM神经网络
- **训练数据**: {CONFIG['EXCEL_FILE']}
- **数据量**: 均衡后{len(pd.read_excel(CONFIG['EXCEL_FILE']))}条 → 增强后{len(pd.read_excel(CONFIG['EXCEL_FILE'])) * (CONFIG['AUGMENT_FACTOR'] + 1)}条
- **输入特征**: 文本序列 + TF-IDF + 选项类型
- **输出类别**: {len(class_names)}个（大类合并后）
- **测试准确率**: {accuracy:.4f}

## 📋 输出类别列表
{chr(10).join([f"- {cls}" for cls in class_names])}

## 🚀 使用步骤
1. 安装依赖包: 
   ```bash
   pip install tensorflow pandas scikit-learn numpy matplotlib imbalanced-learn"""