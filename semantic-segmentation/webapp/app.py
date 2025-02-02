import os
import io
import uuid
import sqlite3
from flask import (
    Flask, request, redirect, url_for, session, flash,
    send_file, render_template_string, send_from_directory, g
)
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import inference
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this for production

# Directory to store uploaded images (organized by user)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# SQLite database filename
DATABASE = 'app.db'

# -----------------------------------------------------------------------------
# Database Helpers
# -----------------------------------------------------------------------------


def get_db():
    """Opens a new database connection if there is none yet for the current application context."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row  # So we can access columns by name.
    return db


@app.teardown_appcontext
def close_connection(exception):
    """Closes the database again at the end of the request."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def init_db():
    """Initialize the database with the required tables and a default admin user."""
    with app.app_context():
        db = get_db()
        # Create the users table.
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
        ''')
        # Create the uploads table.
        db.execute('''
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                original_filename TEXT,
                stored_filename TEXT,
                inference_filename TEXT,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        db.commit()

        # Insert a default admin user if not already present.
        cur = db.execute('SELECT * FROM users WHERE username = ?', ('admin',))
        if cur.fetchone() is None:
            password_hash = generate_password_hash('password')
            db.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                       ('admin', password_hash))
            db.commit()


# Initialize the database when the app starts.
init_db()


def get_user_by_username(username):
    """Retrieve a user row by username."""
    db = get_db()
    cur = db.execute('SELECT * FROM users WHERE username = ?', (username,))
    return cur.fetchone()

# -----------------------------------------------------------------------------
# Dummy Inference Function
# -----------------------------------------------------------------------------


def inference_function(image_path, inference_path):
    """
    A dummy inference function that takes an image file path,
    converts the image to grayscale, and saves the result.
    Replace this with your actual inference/segmentation code.
    """
    # image = Image.open(image_path)
    # segmented_image = ImageOps.grayscale(image)
    # segmented_image.save(inference_path, 'PNG')
    inference.infer(image_path, inference_path)

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('upload'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user_by_username(username)
        # Check if the user exists and if the password matches.
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            session['user_id'] = user['id']
            return redirect(url_for('upload'))
        else:
            error = 'Invalid Credentials. Please try again.'
    return render_template_string('''
    <!doctype html>
    <html>
      <head>
        <title>Login</title>
      </head>
      <body>
        <h1>Login</h1>
        {% if error %}
          <p style="color:red;">{{ error }}</p>
        {% endif %}
        <form method="post">
          <label for="username">Username:</label>
          <input type="text" id="username" name="username"><br><br>
          <label for="password">Password:</label>
          <input type="password" id="password" name="password"><br><br>
          <input type="submit" value="Login">
        </form>
      </body>
    </html>
    ''', error=error)


@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('user_id', None)
    return redirect(url_for('login'))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    user_id = session['user_id']
    # Create a folder for the user if it doesn't exist.
    user_dir = os.path.join(app.config['UPLOAD_FOLDER'], username)
    os.makedirs(user_dir, exist_ok=True)

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        if file:
            # Generate a unique ID for the upload.
            uid = uuid.uuid4().hex
            # Save the original file with a unique name.
            orig_filename = uid + "_" + secure_filename(file.filename)
            orig_path = os.path.join(user_dir, orig_filename)
            file.save(orig_path)

            # Define the inference image filename.
            inference_filename = uid + "_inference.png"
            inference_path = os.path.join(user_dir, inference_filename)
            # Process the image with the dummy inference function.
            inference_function(orig_path, inference_path)

            # Record the upload details in the database.
            db = get_db()
            db.execute('''
                INSERT INTO uploads (user_id, original_filename, stored_filename, inference_filename)
                VALUES (?, ?, ?, ?)
            ''', (user_id, file.filename, orig_filename, inference_filename))
            db.commit()

            flash('Image processed successfully!')
            return redirect(url_for('history'))

    return render_template_string('''
    <!doctype html>
    <html>
      <head>
        <title>Upload Image</title>
      </head>
      <body>
        <h1>Upload an Image for Segmentation</h1>
        <form method="post" enctype="multipart/form-data">
          <input type="file" name="image"><br><br>
          <input type="submit" value="Upload">
        </form>
        <br>
        <a href="{{ url_for('history') }}">View History</a><br>
        <a href="{{ url_for('logout') }}">Logout</a>
      </body>
    </html>
    ''')


@app.route('/uploads/<username>/<filename>')
def uploaded_file(username, filename):
    """Serve a file from the per-user upload folder."""
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], username), filename)


def get_user_uploads(user_id):
    """Retrieve all uploads for a given user ordered by upload time descending."""
    db = get_db()
    cur = db.execute('''
        SELECT * FROM uploads
        WHERE user_id = ?
        ORDER BY upload_time DESC
    ''', (user_id,))
    return cur.fetchall()


@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    user_id = session['user_id']
    uploads = get_user_uploads(user_id)

    history_html = '''
    <!doctype html>
    <html>
      <head>
        <title>Upload History</title>
      </head>
      <body>
        <h1>Upload History for {{ username }}</h1>
        <a href="{{ url_for('upload') }}">Upload New Image</a> | 
        <a href="{{ url_for('logout') }}">Logout</a>
        <hr>
        {% if uploads %}
          {% for upload in uploads %}
            <div style="margin-bottom:20px;">
              <div style="display:inline-block; margin-right:20px;">
                <h3>Original Image ({{ upload.original_filename }})</h3>
                <img src="{{ url_for('uploaded_file', username=username, filename=upload.stored_filename) }}" style="max-width:300px;">
              </div>
              <div style="display:inline-block;">
                <h3>Inference Image</h3>
                <img src="{{ url_for('uploaded_file', username=username, filename=upload.inference_filename) }}" style="max-width:300px;">
              </div>
              <p>Uploaded on: {{ upload.upload_time }}</p>
            </div>
          {% endfor %}
        {% else %}
          <p>No images uploaded yet.</p>
        {% endif %}
      </body>
    </html>
    '''
    return render_template_string(history_html, username=username, uploads=uploads)


if __name__ == '__main__':
    app.run(debug=True)
