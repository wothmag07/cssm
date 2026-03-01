-- =============================================================
-- CSSM — Supabase pgvector Setup
-- Run this in the Supabase SQL Editor (https://supabase.com/dashboard)
-- =============================================================

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
