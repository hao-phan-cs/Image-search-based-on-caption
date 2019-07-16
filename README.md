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
pip3 install numpy
pip3 install opencv-contrib-python
pip3 install scipy matplotlib pillow
pip3 install imutils h5py requests progressbar2
pip3 install scikit-learn scikit-image
pip3 install tensorflow==2.0.0-alpha0
pip3 install tqdm
pip3 install spacy
python -m spacy download en_core_web_sm
pip3 install Flask
pip3 install mysql-connector-python
pip3 install gdown
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
$ python prepare_data.py
```

- generate_model.py: Định nghĩa kiến trúc model

- train_model.py: Training model 
```
$ python train_model.py
```

- eval_model.py: Đánh giá skill model 
```
$ python eval_model.py
```

- gen_caption.py: Sử dụng model để mô tả một ảnh 
```
$ python gen_caption.py
```

## Bài toán truy vấn ảnh
- Sử dụng mô hình Vector Space Model để lập chỉ mục truy vấn hình ảnh
### Mô tả các scripts:
-	sql_statement.py: chứa các hàm khởi tạo cơ sở dữ liệu
-	preprocess_data.py: tiền xử lý dữ liệu
-	indexing.py: đánh chỉ mục trên MySQL
-	run_query.py: thực hiện truy vấn trên cấu trúc chỉ mục

## Running the tests
### Tạo database MySQL trên máy local:
- Vào thư mục indexing
- Download file ir_system3
``` 
$ gdown https://drive.google.com/a/gm.uit.edu.vn/uc?id=11Ui_z6yBe6mBmjkzSh_K737XuzuXfBs_
```
- Setup MySQL server:
```
$ mysql_secure_installation
```
- Đặt password cho root@localhost:
```
$ mysql -u root
mysql> uninstall plugin validate_password;
mysql> ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'test';
mysql> quit
```
- Mở MySQL server với file ir_system3:
```
$ mysql -h localhost -u root -p < ir_system3.sql \
```
- Chạy các scripts:
```
$ python sql_statement.py
$ python indexing.py
```
- Khởi động server flask:
```
$ python main.py
```
- Mở browser và truy cập địa chỉ: http://localhost:5000
- Hoặc truy cập đường link sau để sử dụng: http://192.168.28.11:5000

## Authors

* **Hoàng Đức Lương** - *15520462*
* **Phạm Vũ Hùng** - *15520279*
* **Forked Edition** by Phan Phú Hào
