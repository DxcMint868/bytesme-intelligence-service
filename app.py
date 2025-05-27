from flask import Flask
from routes.product_routes import product_bp
from routes.product_routes import product_bp
from dotenv import load_dotenv

load_dotenv(override=True)

app = Flask(__name__)

# Register blueprints
app.register_blueprint(product_bp)

if __name__ == '__main__':
    app.run(debug=True)
