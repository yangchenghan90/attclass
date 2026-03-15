"""
饮料推荐模型使用示例
展示如何在自己的代码中调用训练好的模型
"""

from predict_model import predict_beverage_option

# ======================== 示例1: 基础使用 ========================
print("=== 示例1: 基础使用 ===")
result1 = predict_beverage_option(
    recommendation_text="5分糖（推荐），5分糖口感佳，3分糖偏淡",
    option_type="甜度"
)
print(f"输入文本: {result1['输入推荐文本']}")
print(f"预测选项: {result1['预测选项']}")
print(f"置信度: {result1['置信度']}")
print()

# ======================== 示例2: 温度预测 ========================
print("=== 示例2: 温度预测 ===")
result2 = predict_beverage_option(
    recommendation_text="正常冰推荐，少冰可选，正常冰口感好",
    option_type="温度"
)
print(f"预测选项: {result2['预测选项']}")
print(f"置信度: {result2['置信度']}")
print()

# ======================== 示例3: 奶茶底预测 ========================
print("=== 示例3: 奶茶底预测 ===")
result3 = predict_beverage_option(
    recommendation_text="经典奶茶底，推荐经典底，浓郁口感",
    option_type="奶茶底"
)
print(f"预测选项: {result3['预测选项']}")
print(f"置信度: {result3['置信度']}")
print()

# ======================== 示例4: 批量预测 ========================
print("=== 示例4: 批量预测 ===")
test_cases = [
    ("3分糖口感淡，3分糖偏酸，三分糖", "甜度"),
    ("少冰推荐，少冰口感佳，少冰可选", "温度"),
    ("全糖推荐，全糖风味更佳，正常糖", "甜度"),
    ("手捣正常冰，正常冰推荐", "温度"),
    ("7分糖推荐，七分糖，7分糖口感", "甜度")
]

print("测试用例 | 预测结果 | 置信度")
print("-" * 50)
for rec_text, opt_type in test_cases:
    result = predict_beverage_option(rec_text, opt_type)
    print(f"{opt_type:<4} | {result['预测选项']:<6} | {result['置信度']:.4f}")

# ======================== 示例5: 结果解析 ========================
print("\n=== 示例5: 详细结果解析 ===")
result = predict_beverage_option("5分糖推荐，五分糖口感好", "甜度")

# 获取所有选项的置信度并排序
sorted_probs = sorted(
    result['所有选项置信度'].items(),
    key=lambda x: x[1],
    reverse=True
)

print("所有选项置信度排名:")
for i, (opt, prob) in enumerate(sorted_probs[:5], 1):
    print(f"  {i}. {opt}: {prob:.4f}")

# 判断结果是否可信
if result['置信度'] >= 0.5:
    print(f"\n✅ 预测结果 '{result['预测选项']}' 可信 (置信度: {result['置信度']})")
else:
    print(f"\n⚠️  预测结果 '{result['预测选项']}' 低置信度，建议人工确认")