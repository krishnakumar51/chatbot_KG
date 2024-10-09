from setuptools import setup, find_packages

setup(
    name='qa_chatbot',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'neo4j',
        'langchain',
        'langchain-core',
        'langchain-community',
        'langchain-huggingface',
        'langchain-groq',
        'pydantic',
        'python-dotenv',
        'flask',
        'flask-cors',
    ],
    entry_points={
        'console_scripts': [
            'qa_chatbot=app:main',  # Replace `app:main` with your actual entry point if needed
        ],
    },
)
