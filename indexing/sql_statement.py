import mysql.connector as mysql
from mysql.connector import Error



def create_new_db(name_db):
    try:
        db = mysql.connect(host='localhost', user='root', passwd='0000')
        cursor = db.cursor()
        cursor.execute("DROP DATABASE IF EXISTS "+name_db)
        cursor.execute("CREATE DATABASE " + name_db)
        print("Successfully create database "+name_db)
    except Error as e:
        print("Error while connecting to MySQL ", e)
    finally:
        if db.is_connected():
            db.close()

def create_table_raw_caption(name_db):
    try:
        db = mysql.connect(host='localhost', user='root', passwd='0000', database=name_db)
        cursor = db.cursor()
        cursor.execute("DROP TABLE IF EXISTS raw_caption")
        query = "CREATE TABLE raw_caption (caption_id int primary key, image_id int, caption varchar(255), norm double)"
        cursor.execute(query)
        print("Successfully create table raw_caption")
    except Error as e:
        print("Error while connecting to MySQL ", e)
    finally:
        if db.is_connected():
            db.close()

def create_table_dictionary(name_db):
    try:
        db = mysql.connect(host='localhost', user='root', passwd='0000', database=name_db)
        cursor = db.cursor()
        cursor.execute("DROP TABLE IF EXISTS dictionary")
        query = "CREATE TABLE dictionary (term varchar(255) primary key, num_of_cap int, idf double)"
        cursor.execute(query)
        print("Successfully create table dictionary")
    except Error as e:
        print("Error while connecting to MySQL ", e)
    finally:
        if db.is_connected():
            db.close()
            
def create_table_posting_list(name_db):   
    try:
        db = mysql.connect(host='localhost', user='root', passwd='0000', database=name_db)
        cursor = db.cursor()
        cursor.execute("DROP TABLE IF EXISTS posting_list")
        query = "CREATE TABLE posting_list (term varchar(255), caption_id int, tf int, weight double, constraint PK_PL primary key (term, caption_id))"
        cursor.execute(query)
        print("Successfully create table posting_list")
    except Error as e:
        print("Error while connecting to MySQL ", e)
    finally:
        if db.is_connected():
            db.close()

def update_weight(name_db):
    try:
        db = mysql.connect(host='localhost', user='root', passwd='0000', database=name_db)
        cursor = db.cursor()
        # calculate tf.idf
        query =  '''UPDATE posting_list as pt, dictionary as d
                    SET pt.weight = pt.tf*d.idf
                    WHERE pt.term = d.term'''
        cursor.execute(query) #54.422s
        db.commit()
        # calculate norm
        query =  '''UPDATE raw_caption as rc, (select pt.caption_id, sqrt(sum(pt.weight*pt.weight)) as norm
					                        from posting_list as pt
                                            group by pt.caption_id) as s
                    SET rc.norm = s.norm
                    WHERE rc.caption_id = s.caption_id'''
        cursor.execute(query) #6.453s
        db.commit()
        # calculate normalized weight
        query =  '''UPDATE posting_list as pt, raw_caption rc
                    SET pt.weight = pt.weight/rc.norm
                    WHERE pt.caption_id = rc.caption_id'''
        cursor.execute(query) #576.938s
        db.commit()
    except Error as e:
        print("Error while connecting to MySQL ", e)
    finally:
        if db.is_connected():
            db.close()

def init_database(name_db):
    create_new_db(name_db)
    create_table_raw_caption(name_db)
    create_table_dictionary(name_db)
    create_table_posting_list(name_db)
    print('Init database successfully')


if __name__ == "__main__":
    create_new_db('ir_system')
    create_table_raw_caption('ir_system')
    create_table_dictionary('ir_system')
    create_table_posting_list('ir_system')
    # insert_value_into_raw_caption('ir_system', 116100, 67, 'A panoramic view of a kitchen and all of its appliances.')
    