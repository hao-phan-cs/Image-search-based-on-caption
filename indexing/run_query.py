import indexing.preprocess_data
import operator
from collections import defaultdict
import mysql.connector as mysql
from mysql.connector import Error
import os

def compute_relevant_score(input_string, name_db, top_k):
    words = indexing.preprocess_data.generate_term(input_string)
    relevant_caps = defaultdict(float)
    try:
        db = mysql.connect(host='localhost', user='root', passwd='0000', database=name_db)
        cursor = db.cursor()
        for word in set(words):
            query = 'SELECT caption_id, weight FROM posting_list WHERE term=\"{}\"'.format(word)
            cursor.execute(query)
            caps_list = cursor.fetchall()
            if len(caps_list)==0:
                continue
            tf_word = words.count(word)
            query = 'SELECT idf FROM dictionary WHERE term=\"{}\"'.format(word)
            cursor.execute(query)
            idf_word = cursor.fetchone()[0]
            w_word = tf_word*idf_word
            for cap_id, w in caps_list:
                relevant_caps[cap_id] = relevant_caps[cap_id] + w*w_word
        sorted_relevant_caps = sorted(relevant_caps.items(), key=operator.itemgetter(1), reverse=True)
        ranked_caps = {}
        scores = []
        for cap_id, score in sorted_relevant_caps:
            query = 'SELECT image_id, caption FROM raw_caption WHERE caption_id={}'.format(cap_id)
            cursor.execute(query)
            image = cursor.fetchone()
            if image[0] not in ranked_caps:
                ranked_caps[image[0]] = image[1]
                scores.append(score)
                if len(ranked_caps)==top_k:
                    return ranked_caps, scores
    except Error as e:
        print("Error while connecting to MySQL ", e)
    finally:
        if db.is_connected():
            db.close()
    return ranked_caps, scores

def query(input_string):
    #input_string = 'A woman and a dog are walking on the beach'
    print('Input: ', input_string)
    ir_system = 'ir_system3' 
    ranked_caps, scores = compute_relevant_score(input_string, ir_system, top_k=20)
    print('Output set with', ir_system)
    print(scores)
    list_img = []
    #image_dir = "../mscoco2014/"
    for image, caption  in ranked_caps.items():
        print(image, ': ', caption)
        list_img.append('COCO_train2014_%012d.jpg' % (image))
    
    return list_img


if __name__ == "__main__":
    lst = query('A woman and a dog are walking on the beach')
    print(lst)

