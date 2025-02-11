from app import app  # Import Flask instance

# Vercel requires this for proper execution
if __name__ == "__main__":
    app.run(debug=True)
