-- CSSM — Supabase pgvector Setup

-- Enable pgvector
create extension if not exists vector;

-- Table for your chunks
create table if not exists public.chunks (
  id bigserial primary key,
  doc_id text not null,
  chunk_index int not null,
  content text not null,
  metadata jsonb default '{}'::jsonb,
  embedding vector(1536)              -- match your embedding dimension
);

-- (Recommended) cosine distance index for fast ANN search
create index if not exists idx_chunks_embedding
  on public.chunks using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- Simple similarity search function (with optional metadata filter)
create or replace function public.match_documents(
  query_embedding vector(1536),
  match_count int default 5,
  filter jsonb default '{}'::jsonb
)
returns table (
  id bigint,
  doc_id text,
  chunk_index int,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
stable
as $$
begin
  return query
  select
    c.id,
    c.doc_id,
    c.chunk_index,
    c.content,
    c.metadata,
    1 - (c.embedding <=> query_embedding) as similarity  -- cosine similarity
  from public.chunks c
  where (filter = '{}'::jsonb) or (c.metadata @> filter)
  order by c.embedding <=> query_embedding               -- lower distance is better
  limit match_count;
end;
$$;


-- =============================================================
-- Hybrid Search (vector + full-text keyword)
-- Computes tsvector inline — no ALTER TABLE needed (free-tier safe)
-- =============================================================

create or replace function public.hybrid_search(
  query_text text,
  query_embedding vector(1536),
  match_count int default 5,
  semantic_weight float default 0.7,
  filter jsonb default '{}'::jsonb
)
returns table (
  id bigint,
  doc_id text,
  chunk_index int,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
stable
as $$
declare
  tsq tsquery := websearch_to_tsquery('english', query_text);
begin
  return query
  select
    sub.id,
    sub.doc_id,
    sub.chunk_index,
    sub.content,
    sub.metadata,
    (
      semantic_weight * sub.vec_score +
      (1 - semantic_weight) * coalesce(
        ts_rank_cd(
          to_tsvector('english',
            coalesce(sub.content, '') || ' ' ||
            coalesce(sub.metadata->>'product_name', '') || ' ' ||
            coalesce(sub.metadata->>'category', '')
          ),
          tsq
        ), 0)
    ) as similarity
  from (
    -- Step 1: fast vector search using ivfflat index
    select
      c.id, c.doc_id, c.chunk_index, c.content, c.metadata,
      1 - (c.embedding <=> query_embedding) as vec_score
    from public.chunks c
    where (filter = '{}'::jsonb) or (c.metadata @> filter)
    order by c.embedding <=> query_embedding
    limit match_count * 4  -- fetch extra candidates for re-ranking
  ) sub
  -- Step 2: re-rank with keyword boost
  order by similarity desc
  limit match_count;
end;
$$;
