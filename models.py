from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import json

# --- CONFIGURARE BAZA DE DATE ---
# Se va crea un fișier 'music.db' automat
SQLALCHEMY_DATABASE_URL = "sqlite:///./music.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# --- DEFINIREA TABELELOR ---

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)

    # Relație: Un user are mai multe preferințe
    preferences = relationship("UserPreference", back_populates="user")


class Song(Base):
    __tablename__ = "songs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, unique=True, index=True)  # Ex: "Metallica - One"
    vector_data = Column(Text)  # Vom stoca vectorul AI ca un string JSON lung

    def get_vector(self):
        """Funcție ajutătoare să primim vectorul ca listă, nu string"""
        return json.loads(self.vector_data)


class UserPreference(Base):
    __tablename__ = "preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    song_id = Column(Integer, ForeignKey("songs.id"))

    user = relationship("User", back_populates="preferences")
    song = relationship("Song")


# Funcție pentru a crea tabelele (o vom apela în backend.py)
def create_tables():
    Base.metadata.create_all(bind=engine)


# Funcție pentru a obține sesiunea de DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()