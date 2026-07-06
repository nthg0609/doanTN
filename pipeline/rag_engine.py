import os
import re
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

class DermatologyRAG:
    def __init__(self, 
                 db_path: str = "5_Results/chroma_db", 
                 corpus_path: str = "9_VQA/medical_guidelines.txt",
                 model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.corpus_path = corpus_path
        
        # Tạo thư mục DB nếu chưa có
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Khởi tạo ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Sử dụng embedding function tích hợp của ChromaDB
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        
        # Khởi tạo collection
        self.collection = self.client.get_or_create_collection(
            name="medical_guidelines",
            embedding_function=self.embedding_fn
        )
        
        # Nạp tài liệu tự động nếu cơ sở dữ liệu trống
        if self.collection.count() == 0:
            self._populate_db()

    def _populate_db(self):
        if not os.path.exists(self.corpus_path):
            print(f"Cảnh báo: Không tìm thấy tệp tài liệu tại {self.corpus_path}")
            return
            
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Tách tài liệu theo các phần [BỆNH LÝ X: ...] bằng re.split
        parts = re.split(r"\[BỆNH LÝ \d+:", content)
        
        documents = []
        metadatas = []
        ids = []
        
        # Bỏ qua phần đầu (header giới thiệu)
        for part in parts[1:]:
            part = part.strip()
            idx = part.find("]")
            if idx != -1:
                header = part[:idx].strip()
                body = part[idx+1:].strip()
                
                # Trích xuất mã bệnh (ví dụ: AKIEC từ "AKIEC - DÀY SỪNG...")
                disease_key = header.split("-")[0].strip().split()[0].strip()
                
                full_text = f"[BỆNH LÝ: {header}]\n{body}"
                
                documents.append(full_text)
                metadatas.append({"disease_key": disease_key})
                ids.append(f"doc_{disease_key}")
            
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Đã lập chỉ mục {len(documents)} phân đoạn tài liệu vào ChromaDB.")

    def retrieve(self, query: str, n_results: int = 1) -> List[Dict[str, Any]]:
        """Tìm kiếm các đoạn tài liệu tương đồng nhất."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            for doc, meta, dist in zip(docs, metas, distances):
                retrieved.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                })
        return retrieved
