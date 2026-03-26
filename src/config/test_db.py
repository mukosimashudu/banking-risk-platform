from src.config.db import engine

def main():
    try:
        with engine.connect() as conn:
            print("✅ Connected to SQL Server!")
    except Exception as e:
        print("❌ Connection failed")
        print(e)

if __name__ == "__main__":
    main()