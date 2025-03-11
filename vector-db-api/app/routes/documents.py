"""API routes for managing documents and their vector embeddings."""

from typing import Optional, List, Dict, Any
import uuid
import logging
import traceback
from fastapi import APIRouter, Depends, HTTPException, Query
from app.models.document import Document, DocumentCreate
from app.services.vector_store import VectorStore
from app.models.search import IndexType
from ..dependencies import get_vector_store


router = APIRouter()


@router.post("/", response_model=Document)
async def create_document(
    document: DocumentCreate,
    vector_store: VectorStore = Depends(get_vector_store),
    index_type: Optional[IndexType] = None,
):
    """Create a new document and store its vector embedding."""
    # Check if this is the first document
    if not vector_store.vector_data:
        logging.info(
            "First document being added. Some indices may not be fully functional until more documents are added."
        )

    doc_id = str(uuid.uuid4())

    try:
        vector_store.add_document(
            doc_id=doc_id,
            content=document.content,
            metadata=document.metadata if hasattr(document, "metadata") else None,
            index_type=index_type,
        )
    except ValueError as e:
        logging.error("ValueError: %s", str(e), exc_info=True)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except (KeyError, AttributeError) as e:
        # Handle data structure errors
        logging.error("Data structure error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=400, detail=f"Invalid data structure: {str(e)}"
        ) from e
    except RuntimeError as e:
        # Handle runtime errors (like embedding or index issues)
        logging.error("Processing error: %s", str(e), exc_info=True)

        # Special handling for IVF training error
        if "Number of training points" in str(
            e
        ) and "should be at least as large as number of clusters" in str(e):
            raise HTTPException(
                status_code=400,
                detail="Not enough documents to train IVF index. Please add more documents or use batch creation with /documents/batch endpoint.",
            ) from e

        raise HTTPException(
            status_code=500, detail=f"Processing error: {str(e)}"
        ) from e
    except Exception as e:  # Still keep this as a last resort
        # For unexpected errors
        logging.error("Unexpected error: %s", str(e), exc_info=True)
        error_detail = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail) from e

    return Document(
        id=doc_id,
        content=document.content,
        metadata=document.metadata if hasattr(document, "metadata") else {},
    )


@router.post("/batch", response_model=List[Document])
async def create_documents_batch(
    documents: List[DocumentCreate],
    vector_store: VectorStore = Depends(get_vector_store),
    index_type: Optional[IndexType] = Query(
        None, description="Index type to use for storing vectors"
    ),
):
    """Create multiple documents in a batch."""
    try:
        # Prepare documents with IDs
        docs_with_ids = []
        for doc in documents:
            doc_id = str(uuid.uuid4())
            docs_with_ids.append(
                {
                    "id": doc_id,
                    "content": doc.content,
                    "metadata": doc.metadata if hasattr(doc, "metadata") else {},
                }
            )

        # Add documents in batch
        vector_store.add_documents_batch(docs_with_ids, index_type)

        # Return created documents
        return [
            Document(id=doc["id"], content=doc["content"], metadata=doc["metadata"])
            for doc in docs_with_ids
        ]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to add documents: {str(e)}"
        ) from e


@router.get("/", response_model=Dict[str, Any])
async def list_documents(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    vector_store: VectorStore = Depends(get_vector_store),
):
    """List documents with pagination."""
    if not vector_store.document_store:
        return {"total": 0, "documents": []}

    # Get document IDs with pagination
    doc_ids = list(vector_store.document_store.keys())[offset : offset + limit]

    # Format documents
    documents = []
    for doc_id in doc_ids:
        doc_data = vector_store.document_store[doc_id]
        documents.append(
            Document(
                id=doc_id,
                content=doc_data["content"],
                metadata=doc_data.get("metadata", {}),
            )
        )

    return {"total": len(vector_store.document_store), "documents": documents}


@router.get("/{document_id}", response_model=Document)
async def get_document(
    document_id: str, vector_store: VectorStore = Depends(get_vector_store)
):
    """Get a document by ID."""
    if document_id not in vector_store.document_store:
        raise HTTPException(status_code=404, detail="Document not found")

    doc_data = vector_store.document_store[document_id]

    return Document(
        id=document_id,
        content=doc_data["content"],
        metadata=doc_data.get("metadata", {}),
    )


@router.post("/test-data", response_model=Dict[str, Any])
async def add_test_data(
    count: int = Query(100, description="Number of test documents to add"),
    category: str = Query(
        "mixed",
        description="Category of test data: 'tech', 'science', 'business', 'mixed'",
    ),
    vector_store: VectorStore = Depends(get_vector_store),
):
    """Add diverse test data to train indices properly."""
    # Define categories of content for more diverse embeddings
    tech_content = [
        "Machine learning algorithms can process vast amounts of data to identify patterns.",
        "Cloud computing enables businesses to scale their IT infrastructure on demand.",
        "Blockchain technology provides a secure and transparent way to record transactions.",
        "Artificial intelligence is transforming how we interact with technology daily.",
        "Quantum computing promises to solve problems that are currently intractable.",
        "Edge computing brings processing power closer to where data is generated.",
        "Natural language processing allows computers to understand human language.",
        "Computer vision systems can identify objects and people in images and videos.",
        "The Internet of Things connects everyday devices to the internet.",
        "Cybersecurity measures protect systems from unauthorized access and attacks.",
        "Virtual reality creates immersive digital environments for users to interact with.",
        "Augmented reality overlays digital information onto the physical world.",
        "5G networks provide faster and more reliable wireless communication.",
        "Big data analytics extracts valuable insights from large datasets.",
        "Robotics combines mechanical engineering with computer science.",
        "DevOps practices integrate software development and IT operations.",
        "Microservices architecture breaks applications into smaller, independent services.",
        "Containerization technology packages applications with their dependencies.",
        "APIs enable different software applications to communicate with each other.",
        "Serverless computing allows developers to build applications without managing servers.",
    ]

    science_content = [
        "The theory of relativity describes the relationship between space and time.",
        "Quantum mechanics explains the behavior of matter at the subatomic level.",
        "DNA carries the genetic instructions for the development of all living organisms.",
        "The periodic table organizes chemical elements based on their properties.",
        "Evolution by natural selection explains how species adapt over time.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "The Big Bang theory describes the origin and evolution of the universe.",
        "Plate tectonics explains the movement of the Earth's lithosphere.",
        "Neurons transmit information through electrical and chemical signals.",
        "Antibiotics kill or inhibit the growth of bacteria that cause infections.",
        "Climate change is altering global weather patterns and ecosystems.",
        "The scientific method provides a framework for conducting research.",
        "Vaccines stimulate the immune system to protect against specific diseases.",
        "Black holes are regions of spacetime where gravity is so strong nothing can escape.",
        "Ecosystems consist of living organisms and their physical environment.",
        "Thermodynamics governs the behavior of energy and its transformations.",
        "The human genome contains all the genetic information of an individual.",
        "Particle accelerators allow scientists to study subatomic particles.",
        "Stem cells have the potential to develop into many different cell types.",
        "The Doppler effect explains how wave frequencies change relative to an observer.",
    ]

    business_content = [
        "Market research helps companies understand their target audience.",
        "Supply chain management optimizes the flow of goods and services.",
        "Strategic planning sets long-term goals and action plans for organizations.",
        "Financial analysis evaluates the viability and profitability of businesses.",
        "Customer relationship management maintains connections with clients.",
        "Digital marketing promotes products and services through online channels.",
        "Human resources management oversees employee recruitment and development.",
        "Operations management ensures efficient production of goods and services.",
        "Risk management identifies and mitigates potential threats to a business.",
        "Corporate social responsibility addresses a company's impact on society.",
        "Mergers and acquisitions combine or purchase companies to create value.",
        "Entrepreneurship involves starting and running new business ventures.",
        "Business intelligence uses data analysis to inform strategic decisions.",
        "Brand management builds and maintains a company's reputation and image.",
        "E-commerce enables buying and selling of goods and services online.",
        "Project management coordinates resources to achieve specific objectives.",
        "Venture capital provides funding for early-stage companies with growth potential.",
        "Intellectual property protects creations of the mind, such as inventions.",
        "Organizational behavior studies how people interact within organizations.",
        "Quality management ensures products and services meet specified standards.",
    ]

    # Generate test documents with varied content
    test_docs = []
    for i in range(count):
        # Select content based on category
        if category == "tech":
            content_list = tech_content
        elif category == "science":
            content_list = science_content
        elif category == "business":
            content_list = business_content
        else:  # mixed
            # Combine all categories and select randomly
            content_list = tech_content + science_content + business_content

        # Select a random content from the appropriate list
        import random

        base_content = random.choice(content_list)

        # Add some variation to make each document unique
        content = f"{base_content} This is additional context for document {i}."

        test_docs.append(
            {
                "id": str(uuid.uuid4()),
                "content": content,
                "metadata": {
                    "test": True,
                    "index": i,
                    "category": category,
                    "length": len(content),
                },
            }
        )

    # Add in batch
    try:
        doc_ids = vector_store.add_documents_batch(test_docs)
        return {
            "message": f"Successfully added {len(doc_ids)} test documents in category '{category}'",
            "count": len(doc_ids),
            "category": category,
        }
    except Exception as e:
        logging.error("Failed to add test data: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to add test data: {str(e)}"
        ) from e


@router.post("/test-data-all", response_model=Dict[str, Any])
async def add_comprehensive_test_data(
    count_per_category: int = Query(
        50, description="Number of test documents to add per category"
    ),
    vector_store: VectorStore = Depends(get_vector_store),
):
    """Add comprehensive test data from all categories to properly train indices."""
    categories = ["tech", "science", "business"]
    results = {}
    total_count = 0

    for category in categories:
        try:
            # Call the regular test data endpoint for each category
            response = await add_test_data(
                count=count_per_category, category=category, vector_store=vector_store
            )
            results[category] = response
            total_count += response["count"]
        except Exception as e:
            results[category] = {"error": str(e)}

    return {
        "message": f"Added {total_count} test documents across all categories",
        "total_count": total_count,
        "category_results": results,
    }
