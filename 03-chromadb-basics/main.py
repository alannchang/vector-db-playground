from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2 # this is how you import in Chroma 0.5.0+
# from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2 # legacy import for Chroma <=0.4.24

ef = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
import chromadb
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="my_collection", embedding_function=ef)

collection.add(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts=["This is a query document about florida"], # Chroma will embed this for you
    n_results=2 # how many results to return
)
print(results)
