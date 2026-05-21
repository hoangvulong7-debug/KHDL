## Hàm đánh giá mô hình

Sau khi các mô hình dự đoán xong, dự án sử dụng hàm `evaluate_model()` để đánh giá chất lượng dự đoán của từng mô hình. Hàm này nhận vào nhãn thực tế `y_true`, nhãn dự đoán `y_pred` và xác suất dự đoán lớp TĂNG `y_prob`.

Trong bài toán này, nhãn được quy ước như sau:

- `0`: GIẢM
- `1`: TĂNG

Hàm đánh giá trả về nhiều chỉ số khác nhau để so sánh mô hình một cách công bằng, thay vì chỉ dựa vào Accuracy.

### ROC-AUC an toàn

```python
def safe_roc_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan
