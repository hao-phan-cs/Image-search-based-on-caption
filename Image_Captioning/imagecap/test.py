import pickle

img_name_vector = pickle.load(open('/home/mmlab/image_captioning/models/img_name_vector.pkl', 'rb'))
print("img_name_vector: ", len(img_name_vector))
encode_train = sorted(set(img_name_vector))
print("encode_train: ", len(encode_train))

img_names_test = pickle.load(open('/home/mmlab/image_captioning/models/img_names_test.pkl', 'rb'))
captions_test = pickle.load(open('/home/mmlab/image_captioning/models/captions_test.pkl', 'rb'))
print("img_names_test: ", len(img_names_test))
print("captions_test: ", len(captions_test)) 