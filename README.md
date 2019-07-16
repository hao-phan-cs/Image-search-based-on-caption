# Image Search Engine Based On Caption
- Đồ án môn học: Xử lý ảnh và ứng dụng (HK2 2018-2019)
- Tên đề tài: Xây dựng hệ thống tìm kiếm ảnh sử dụng mô tả ( Image search engine based on caption )
  + Input: Một câu mô tả ảnh hoặc một hình ảnh
  + Output: Các hình ảnh liên quan với câu mô tả
  
## Ngôn ngữ sử dụng
- python 3
- Thư viện Tensorflow
- Thư viện Flask

## Cài đặt các thư viện cần thiết
```
pip install numpy
pip install opencv-contrib-python
pip install scipy matplotlib pillow
pip install imutils h5py requests progressbar2
pip install scikit-learn scikit-image
pip install tensorflow==2.0.0-alpha0
pip install tqdm
pip install spacy
python -m spacy download en_core_web_sm
pip install Flask
pip install mysql-connector-python
```

## Bài toán image captioning
- Training một model có khả năng tạo ra câu mô tả cho hình ảnh đầu vào

### Download the MS-COCO dataset
- annotations: http://images.cocodataset.org/annotations/annotations_trainval2014.zip 
- images: http://images.cocodataset.org/zips/train2014.zip

### Mô tả các thư mục
- annotations: Chứa các file .json được giải nén từ tập tin annotations_trainval2014.zip
- mscoco2014: Chứa các hình ảnh được giải nén từ tập tin train2014.zip
- features_incepv3: Chứa các features của các hình ảnh dùng cho việc training model
- models: Chứa các dữ liệu được tạo ra khi chạy các file scripts
- models/checkpoints: Chứa các Checkpoints lưu các parameters của model

### Mô tả các scripts
- prepare_data.py: Chuẩn bị dữ liệu cho training 
```
run: python prepare_data.py
```

- generate_model.py: Định nghĩa kiến trúc model

- train_model.py: Training model 
```
run: python train_model.py
```

- eval_model.py: Đánh giá skill model 
```
run: python eval_model.py
```

- gen_caption.py: Sử dụng model để mô tả một ảnh 
```
run: python gen_caption.py
```

## Bài toán truy vấn ảnh
- Sử dụng mô hình Vector Space Model để lập chỉ mục truy vấn hình ảnh
### Mô tả các scripts:
-	sql_statement.py: chứa các hàm khởi tạo cơ sở dữ liệu
```
run: python sql_statement.py
```

-	preprocess_data.py: tiền xử lý dữ liệu

-	indexing.py: đánh chỉ mục trên MySQL
```
run: python indexing.py
```

-	run_query.py: thực hiện truy vấn trên cấu trúc chỉ mục
```
run: python run_query.py
```

## Running the tests

- Download file ir_system3 vào thư m� indexing https://drive.google.com/a/gm.uit.edu.vn/uc?id=11Ui_z6yBe6mBmjkzSh_K737XuzuXfBs_
- Chạy trên local:
```
1. Tạo database MySQL trên máy local bằng cách thực thi file ir_system3.sql trong thư mục indexing:
  - Click chuột phải vào file ir_system3.sql ==> Open with ==> MySQL WorkBench ==> Chọn Run SQL Scripts...
2. run: python main.py
3. Mở browser và truy cập địa chỉ: http://localhost:5000
```

- Hoặc truy cập đường link sau để sử dụng:
```
http://35.232.224.165
```
## Authors

* **Hoàng Đức Lương** - *15520462*
* **Phạm Vũ Hùng** - *15520279*
