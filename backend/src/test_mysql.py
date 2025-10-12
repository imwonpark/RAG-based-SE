import pymysql
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../../.env')

try:
    # Try to connect
    connection = pymysql.connect(
        host=os.getenv('MYSQL_HOST', 'localhost'),
        user=os.getenv('MYSQL_USER', 'root'),
        password=os.getenv('MYSQL_PASSWORD', ''),
        database=os.getenv('MYSQL_DATABASE', 'rag_search'),
        port=int(os.getenv('MYSQL_PORT', 3306))
    )
    
    print("✅ MySQL connection successful!")
    
    # Test query
    with connection.cursor() as cursor:
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()
        print(f"✅ Found {len(tables)} tables:")
        for table in tables:
            print(f"   - {table[0]}")
    
    connection.close()
    print("\n🎉 MySQL is ready to use!")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure MySQL is running: brew services list")
    print("2. Check your password in .env file")
    print("3. Verify database exists: mysql -u root -p -e 'SHOW DATABASES;'")
