from typing import Dict, Any
import time
import logging
from fastapi import APIRouter, Depends, HTTPException

from app.models.search import SearchRequest, SearchResponse, SearchResult
from app.services.vector_store import VectorStore
from app.config.settings import settings
from ..dependencies import get_vector_store

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest, vector_store: VectorStore = Depends(get_vector_store)
):
    """Search for similar documents using vector similarity"""
    start_time = time.time()

    try:
        # Perform search
        results, used_index_type = vector_store.search(
            query=request.query,
            k=request.k,
            index_type=request.index_type,
            search_params=request.params,
        )

        # Format response
        search_results = [
            SearchResult(
                document_id=r["document_id"],
                content=r["content"],
                distance=r["distance"],
                metadata=r["metadata"],
            )
            for r in results
        ]

        search_time = time.time() - start_time
        logging.info(
            "Search with %s completed in %.4f seconds", used_index_type, search_time
        )

        return SearchResponse(
            query=request.query,
            results=search_results,
            index_type=used_index_type,
            search_time_ms=search_time * 1000,
            total_results=len(results),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except (KeyError, IndexError) as e:
        # Handle data structure errors
        logging.error("Data structure error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Data structure error: {str(e)}"
        ) from e
    except RuntimeError as e:
        # Handle runtime errors (like FAISS issues)
        logging.error("Search engine error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Search engine error: {str(e)}"
        ) from e
    except Exception as e:  # Still keep this as a last resort
        logging.error("Unexpected error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}"
        ) from e


@router.post("/benchmark", response_model=Dict[str, Any])
async def benchmark_search(
    request: SearchRequest, vector_store: VectorStore = Depends(get_vector_store)
):
    """Benchmark search across all available index types"""
    if not vector_store.vector_data:
        raise HTTPException(status_code=400, detail="No documents in the database")

    results = {}

    # Test each enabled index type
    for index_type in settings.ENABLED_INDICES:
        try:
            start_time = time.time()

            # Perform search with this index type
            search_results, used_index_type = vector_store.search(
                query=request.query,
                k=request.k,
                index_type=index_type,
                search_params=request.params,
            )

            search_time = time.time() - start_time

            # Record results
            results[index_type] = {
                "search_time_ms": search_time * 1000,
                "results_count": len(search_results),
                "used_index_type": used_index_type,  # This might differ if fallback occurred
                "top_result": search_results[0]["content"] if search_results else None,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except (KeyError, AttributeError) as e:
            # Handle data structure errors
            raise HTTPException(
                status_code=400, detail=f"Invalid data structure: {str(e)}"
            ) from e
        except RuntimeError as e:
            # Handle runtime errors (like embedding or index issues)
            raise HTTPException(
                status_code=500, detail=f"Processing error: {str(e)}"
            ) from e
        except Exception as e:  # Still keep this as a last resort
            # For unexpected errors
            raise HTTPException(
                status_code=500, detail=f"Unexpected error: {str(e)}"
            ) from e

    return {"query": request.query, "benchmark_results": results}
