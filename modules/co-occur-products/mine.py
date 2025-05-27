import psycopg2
import pandas as pd
import sqlite3
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from collections import defaultdict
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# 1. Kết nối CSDL PostgreSQL
conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
    user=os.getenv("POSTGRES_USERNAME"),
    password=os.getenv("POSTGRES_PASSWORD"),
    dbname=os.getenv("POSTGRES_DATABASE"),
)

# Truy vấn đơn hàng
query = """
SELECT order_id, ARRAY_AGG(product_id ORDER BY product_id) AS products
FROM order_items
GROUP BY order_id;
"""
df = pd.read_sql_query(query, conn)

# Tạo transactions
transactions = df["products"].tolist()
transactions = [t for t in transactions if len(t) > 1]
print(f"✅ Transactions sau khi lọc (length > 1): {len(transactions)}")

# Encode dữ liệu
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_tf = pd.DataFrame(te_ary, columns=te.columns_)

# FP-Growth với min_support thấp
freq_items = fpgrowth(df_tf, min_support=0.0001, use_colnames=True)
freq_items['length'] = freq_items['itemsets'].apply(lambda x: len(x))
print(f"✅ Số itemsets ≥ 2: {len(freq_items)}")

# Sinh luật
rules = association_rules(freq_items, metric="confidence", min_threshold=0.1)
print(f"✅ Số luật sinh được: {len(rules)}")

# Tạm thời lưu [(consequent, confidence, support)] cho từng product
temp_map = defaultdict(list)

for _, row in rules.iterrows():
    antecedents = list(row['antecedents'])
    consequents = list(row['consequents'])
    confidence = row['confidence']
    support = row['support']

    for a in antecedents:
        for c in consequents:
            temp_map[a].append((c, confidence, support))

# Với mỗi product, giữ lại những related_product có độ tin cậy cao nhất
related_map = {}
for a, related_list in temp_map.items():
    # Sắp xếp theo confidence giảm dần, nếu bằng thì theo support giảm dần
    sorted_list = sorted(related_list, key=lambda x: (-x[1], -x[2]))
    # Giữ lại 1 sản phẩm liên quan tốt nhất
    top_related = [x[0] for x in sorted_list[:1]]
    related_map[str(a)] = list(map(int, top_related))

# Liên kết với CSDL SQLITE3
# Kết nối SQLite (tạo file nếu chưa có)
conn_sqlite = sqlite3.connect("scripts/co-occur-products/co-occur-products.db")
cur = conn_sqlite.cursor()

# Tạo bảng nếu chưa có
cur.execute("DROP TABLE IF EXISTS related_products")
cur.execute("""
    CREATE TABLE related_products (
        product_id TEXT,
        related_id TEXT
    )
""")

# Ghi từng cặp vào bảng
for pid, related_list in related_map.items():
    for rid in related_list:
        cur.execute(
            "INSERT INTO related_products (product_id, related_id) VALUES (?, ?)", (pid, str(rid)))

# Lưu và đóng kết nối
conn_sqlite.commit()
conn_sqlite.close()

print(
    f"✅ Đã ghi {sum(len(v) for v in related_map.values())} bản ghi vào related_products.db")
