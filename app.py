
from flask import Flask, render_template,redirect,url_for, request,flash,session
from sqlalchemy import true
from flask_mysqldb import MySQL
import re 
import yaml
import datetime
app= Flask(__name__)
app.secret_key="Secret@11"
#Configure db
db = yaml.load(open('db.yaml'),Loader=yaml.Loader)
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']

mysql = MySQL(app)

#Sentimental analysis
def sentiment(blog):
        final_list_pos = []
        count_pos=0
        positive_words = open('positive-words.txt',"r")
        list_pos = positive_words.readlines()
        for i in list_pos:
            final_list_pos.append(i.strip())
        sentence = blog.lower()    
        sentence = sentence.split()
        for i in  final_list_pos:
            for words in sentence:
                if i in words:
                    count_pos+=1
        print(count_pos)
        final_list_neg = []
        count_neg=0
        negative_words = open('negative-words.txt',"r")
        list_neg = negative_words.readlines()
        for i in list_neg:
             final_list_neg.append(i.strip())
        sentence = blog.lower()       
        sentence = blog.split()
        for i in  final_list_neg:
            for words in sentence:
                if i in words:
                    count_neg+=1
        print(count_neg)
        sentiment = count_pos - count_neg
        if sentiment < 0:
            sentiment = "negative"
        elif sentiment == 0: 
            sentiment = "neutral"
        else:
            sentiment ="positive"
        print (sentiment)
        return sentiment

#Censoring
def censor(sentence):
        badwords = open('wordlist.txt')
        lines=[]
        for line in badwords:
            lines.append(line.strip())
        sentence = sentence.split()
        for index, word in enumerate(sentence):
            if any(badword in word for badword in lines):
                sentence[index] = "".join(['*' if c.isalpha() else c for c in word])
        result= " ".join(sentence)
        print (result)
        return result

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/sign-up',methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        #fetch details from form
        userDetails = request.form
        user_name = userDetails['username']
        email = userDetails['email']
        gender = userDetails['gender']
        college = userDetails['college']
        branch = userDetails['branch']
        yos = userDetails['yos']
        password = userDetails['password']
        cpass = userDetails['cpassword']
        cur = mysql.connection.cursor()
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM STUDENT WHERE EMAIL = %s', (email,))
        account = cursor.fetchone()
        if password == cpass:
         # If account exists show error and validation checks
         if account:
             msg = 'Account already exists!'
         elif not re.match(r'[A-Za-z]+', user_name):
             msg = 'Username must contain only characters '
         else:
             cur.execute("INSERT INTO STUDENT(name,email,gender,college,branch,year_of_study,password) VALUES(%s,%s,%s,%s,%s,%s,%s) ",(user_name,email,gender,college,branch,yos,password))
             mysql.connection.commit()
             cur.close()
             msg='Success'
        else:
         msg = 'Passwords did not match'
        flash(msg) 
        return render_template('sign up.html') 
    return render_template('sign up.html')  

@app.route('/login',methods=['GET','POST'])
def login():
    return render_template('login.html')  

@app.route('/login-validate',methods=['GET','POST'])
def loginvalidate():
    if request.method == 'POST':
        #fetch details from form
        userDetails = request.form
        email = userDetails['email']
        password = userDetails['password']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM STUDENT WHERE EMAIL = %s AND PASSWORD=%s ",(email,password))
        account=cur.fetchone()
         # If account exists in accounts table in out database
        if account:
            session['loggedin'] = True
            session['id'] = account[1]
            session['username'] = account[0]
            return redirect(url_for('student'))
        else:
            # Account doesnt exist or username/password incorrect
            return render_template('login.html')
        mysql.connection.commit()
        cur.close()
        
@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))

@app.route('/student')
def student():
    if 'loggedin' in session:
             # User is loggedin show them the dashboard page
              return render_template('student.html', username=session['username'],email=session['id'])
    else:          
      return redirect(url_for('login'))

@app.route('/admin')
def admin():
  return render_template('admin.html')

@app.route('/micro-blogging')
def blog():
    if 'loggedin' in session:
            # User is loggedin show them the page
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM BLOG")
            data=cur.fetchall()
            return render_template('micro blogging.html',value=data, username=session['username'],email=session['id'])
    else:          
      return redirect(url_for('login'))
@app.route('/calendar')
def calendar():
    if 'loggedin' in session:
        return render_template('academic calendar.html', username=session['username'],email=session['id'])
    else:          
      return redirect(url_for('login'))
@app.route('/books')
def book():
    if 'loggedin' in session:
             # User is loggedin show them the page
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM BOOKS")
            data=cur.fetchall()
            return render_template('book recommender.html',value=data, username=session['username'],email=session['id'])
    else:          
      return redirect(url_for('login'))
    

@app.route('/addpost',methods=['POST','GET'])
def add_post():
    #fetch details from form
    postDetails = request.form
    post_data = postDetails['area']
    post_type = postDetails['post_type']
    posted_by = session['username']
    posted_on=datetime.datetime.now()
    #Find out sentiment
    post_sentiment=sentiment(post_data)
    #Apply censoring
    post_data=censor(post_data)
    #Insert post details in database
    cursor = mysql.connection.cursor()
    cursor.execute("INSERT INTO BLOG(BLOG_TEXT,POSTED_BY,POSTED_ON,SENTIMENT,POST_TYPE) VALUES(%s,%s,%s,%s,%s) ",(post_data,posted_by,posted_on,post_sentiment,post_type))
    mysql.connection.commit()
    cursor.close()
    return redirect(url_for('student'))

@app.route('/show-posts',methods=['POST','GET'])
def show_posts():
    postDetails = request.form
    post_type = postDetails['post_type']
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM BLOG WHERE POST_TYPE=%s",(post_type))
    data=cur.fetchall()
    return render_template('micro blogging.html',value=data, username=session['username'],email=session['id'])
    
if __name__ == '__main__': 
    app.run(debug=true)
