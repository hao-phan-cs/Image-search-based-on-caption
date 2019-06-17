import json
import math
from collections import defaultdict
from collections import Counter
import sql_statement
import preprocess_data
import mysql.connector as mysql
from mysql.connector import Error
from tqdm import tqdm


def load_collection_and_store_data(annotation_file, name_db):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    collection = defaultdict(list)
    try:
        db = mysql.connect(host='localhost', user='root', passwd='1234', database=name_db)
        # db.autocommit = True
        cursor = db.cursor()
        idf_counter = Counter()
        for annot in tqdm(annotations['annotations']):
            if '"' in annot['caption']:
                continue
            collection[annot['image_id']].append({'id':annot['id'], 'caption':annot['caption']})
            query = "INSERT INTO raw_caption (caption_id, image_id, caption) VALUES ({}, {}, \"{}\")".format(annot['id'], annot['image_id'], annot['caption'])
            cursor.execute(query)
            term_in_doc = preprocess_data.generate_term(annot['caption'])
            idf_counter.update(set(term_in_doc))
            tf_counter = Counter(term_in_doc)
            for term in tf_counter.keys():
                query = "INSERT INTO posting_list (term, caption_id, tf) VALUES (\"{}\",{},{})".format(term, annot['id'], tf_counter[term])
                cursor.execute(query)
        for term in tqdm(idf_counter.keys()):
            num_of_cap = idf_counter[term]
            idf = 1 + math.log10(len(idf_counter)/float(num_of_cap))
            query = "INSERT INTO dictionary (term, num_of_cap, idf) VALUES (\"{}\",{},{})".format(term, num_of_cap, idf)
            cursor.execute(query)
        db.commit()
    except Error as e:
        print("Error while connecting to MySQL ", e)
    finally:
        if db.is_connected():
            db.close()
    
    return collection
    

def image_path(image_id, image_dir):
    return image_dir + 'COCO_train2014_' + '%012d.jpg' % (image_id)

if __name__ == "__main__":
    annotation_file = '../annotations/captions_train2014.json'
    image_dir = '../mscoco2014/'
    sql_statement.init_database('ir_system3')
    collection = load_collection_and_store_data(annotation_file, 'ir_system3')
    json.dump(collection, open('collection.json', 'w'), indent=4)
    sql_statement.update_weight('ir_system3')
    # Retain 82782 images, 413068 captions
    
    