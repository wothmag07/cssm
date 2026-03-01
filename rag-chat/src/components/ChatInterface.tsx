'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, Sparkles, Zap, ChevronDown, ChevronUp, Star } from 'lucide-react';
import { playThinkingSound, playResponseSound } from '@/utils/sounds';

interface Source {
  content: string;
  metadata: {
    product_name?: string;
    rating?: number;
    avg_rating?: number;
    category?: string;
    store?: string;
    price?: string;
  };
  similarity: number;
}

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'bot';
  timestamp: string;
  sources?: Source[];
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const toggleSources = (messageId: string) => {
    setExpandedSources((prev) => {
      const next = new Set(prev);
      if (next.has(messageId)) next.delete(messageId);
      else next.add(messageId);
      return next;
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    if (showWelcome) setShowWelcome(false);

    const userMessage: Message = {
      id: Date.now().toString(),
      content: input.trim(),
      sender: 'user',
      timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    playThinkingSound();

    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || `http://${window.location.hostname}:8001`;
      const response = await fetch(`${baseUrl}/retrieve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ msg: userMessage.content }),
      });

      const data = await response.json();

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.response || 'Error: ' + (data.error || 'Unknown error'),
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        sources: data.sources || [],
      };

      setMessages((prev) => [...prev, botMessage]);
      playResponseSound();
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'Connection failed. Make sure the backend server is running on port 8001.',
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestion = (text: string) => {
    setInput(text);
    inputRef.current?.focus();
  };

  const formatContent = (text: string) => {
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong class="text-[#e8e8e8] font-semibold">$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/\n/g, '<br>');
  };

  return (
    <div className="flex flex-col h-screen" style={{ background: '#0a0a0a' }}>
      {/* ── Header ── */}
      <header
        className="flex items-center justify-between px-6 py-4"
        style={{ borderBottom: '1px solid #1a1a1a' }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: '#00d4aa' }}
          >
            <Zap className="w-4 h-4" style={{ color: '#0a0a0a' }} />
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-wide" style={{ color: '#e8e8e8' }}>
              CSSM
            </h1>
            <p className="text-[11px]" style={{ color: '#555' }}>
              Customer Support System Management
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: '#00d4aa' }} />
          <span className="text-[11px] font-medium" style={{ color: '#555' }}>Online</span>
        </div>
      </header>

      {/* ── Chat Area ── */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-2xl mx-auto space-y-4">
          {/* Welcome Card */}
          {showWelcome && (
            <div className="animate-fadeInUp" style={{ marginTop: '10vh' }}>
              <div
                className="rounded-xl p-6"
                style={{ background: '#141414', border: '1px solid #222' }}
              >
                <div className="flex items-center gap-2 mb-4">
                  <Sparkles className="w-4 h-4" style={{ color: '#00d4aa' }} />
                  <span className="text-sm font-medium" style={{ color: '#e8e8e8' }}>
                    Product Assistant
                  </span>
                </div>
                <p className="text-[13px] leading-relaxed mb-5" style={{ color: '#888' }}>
                  Ask about electronics products. I analyze real customer reviews from Amazon
                  to give you honest, data-driven recommendations.
                </p>
                <div className="flex flex-wrap gap-2">
                  {[
                    'Best budget laptops for students',
                    'Compare AirPods vs Galaxy Buds',
                    'Top rated 4K monitors under $500',
                  ].map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => handleSuggestion(suggestion)}
                      className="text-[12px] px-3 py-1.5 rounded-lg transition-all duration-200"
                      style={{
                        background: '#1a1a1a',
                        border: '1px solid #2a2a2a',
                        color: '#888',
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.borderColor = '#00d4aa';
                        e.currentTarget.style.color = '#00d4aa';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.borderColor = '#2a2a2a';
                        e.currentTarget.style.color = '#888';
                      }}
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Messages */}
          {messages.map((message) => (
            <div key={message.id} className="animate-fadeInUp">
              <div
                className={`flex gap-3 ${
                  message.sender === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                {message.sender === 'bot' && (
                  <div
                    className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 mt-1"
                    style={{ background: '#1e1e1e', border: '1px solid #2a2a2a' }}
                  >
                    <Bot className="w-3.5 h-3.5" style={{ color: '#00d4aa' }} />
                  </div>
                )}

                <div
                  className="max-w-[75%] rounded-2xl px-4 py-3"
                  style={
                    message.sender === 'user'
                      ? { background: '#00d4aa', color: '#0a0a0a' }
                      : { background: '#141414', border: '1px solid #1e1e1e', color: '#d0d0d0' }
                  }
                >
                  <div
                    className="text-[13px] leading-relaxed whitespace-pre-wrap break-words"
                    dangerouslySetInnerHTML={{ __html: formatContent(message.content) }}
                  />
                  <div className="flex items-center justify-between mt-2">
                    <span
                      className="text-[10px]"
                      style={{ color: message.sender === 'user' ? 'rgba(10,10,10,0.5)' : '#444' }}
                    >
                      {message.timestamp}
                    </span>
                    {/* Sources toggle button */}
                    {message.sender === 'bot' && message.sources && message.sources.length > 0 && (
                      <button
                        onClick={() => toggleSources(message.id)}
                        className="flex items-center gap-1 text-[10px] transition-colors duration-200"
                        style={{ color: '#00d4aa' }}
                      >
                        {expandedSources.has(message.id) ? (
                          <>
                            <ChevronUp className="w-3 h-3" />
                            Hide sources
                          </>
                        ) : (
                          <>
                            <ChevronDown className="w-3 h-3" />
                            {message.sources.length} sources
                          </>
                        )}
                      </button>
                    )}
                  </div>
                </div>

                {message.sender === 'user' && (
                  <div
                    className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 mt-1"
                    style={{ background: '#1e1e1e', border: '1px solid #2a2a2a' }}
                  >
                    <User className="w-3.5 h-3.5" style={{ color: '#888' }} />
                  </div>
                )}
              </div>

              {/* Expanded Source Citations */}
              {message.sender === 'bot' &&
                message.sources &&
                expandedSources.has(message.id) && (
                  <div className="ml-10 mt-2 space-y-1.5 animate-fadeInUp">
                    {message.sources.map((source, idx) => (
                      <div
                        key={idx}
                        className="rounded-lg px-3 py-2 text-[11px]"
                        style={{
                          background: '#111',
                          border: '1px solid #1e1e1e',
                        }}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-medium" style={{ color: '#aaa' }}>
                            [{idx + 1}] {source.metadata.product_name || 'Unknown Product'}
                          </span>
                          <div className="flex items-center gap-2">
                            {source.metadata.rating && (
                              <span className="flex items-center gap-0.5" style={{ color: '#f0b429' }}>
                                <Star className="w-2.5 h-2.5 fill-current" />
                                {source.metadata.rating}
                              </span>
                            )}
                            <span
                              className="px-1.5 py-0.5 rounded text-[9px] font-medium"
                              style={{
                                background:
                                  source.similarity > 0.8
                                    ? 'rgba(0,212,170,0.15)'
                                    : source.similarity > 0.6
                                      ? 'rgba(240,180,41,0.15)'
                                      : 'rgba(255,100,100,0.15)',
                                color:
                                  source.similarity > 0.8
                                    ? '#00d4aa'
                                    : source.similarity > 0.6
                                      ? '#f0b429'
                                      : '#ff6464',
                              }}
                            >
                              {Math.round(source.similarity * 100)}% match
                            </span>
                          </div>
                        </div>
                        {source.metadata.category && (
                          <span className="text-[10px]" style={{ color: '#555' }}>
                            {source.metadata.category}
                            {source.metadata.price ? ` · ${source.metadata.price}` : ''}
                          </span>
                        )}
                        <p className="mt-1" style={{ color: '#666' }}>
                          {source.content}
                          {source.content.length >= 200 ? '...' : ''}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
            </div>
          ))}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex gap-3 justify-start animate-fadeInUp">
              <div
                className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 mt-1 glow-pulse"
                style={{ background: '#1e1e1e', border: '1px solid #2a2a2a' }}
              >
                <Bot className="w-3.5 h-3.5" style={{ color: '#00d4aa' }} />
              </div>
              <div
                className="rounded-2xl px-4 py-3"
                style={{ background: '#141414', border: '1px solid #1e1e1e' }}
              >
                <div className="flex items-center gap-1.5">
                  <div
                    className="w-1.5 h-1.5 rounded-full dot-pulse"
                    style={{ background: '#00d4aa', animationDelay: '0s' }}
                  />
                  <div
                    className="w-1.5 h-1.5 rounded-full dot-pulse"
                    style={{ background: '#00d4aa', animationDelay: '0.2s' }}
                  />
                  <div
                    className="w-1.5 h-1.5 rounded-full dot-pulse"
                    style={{ background: '#00d4aa', animationDelay: '0.4s' }}
                  />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* ── Input Area ── */}
      <div className="px-4 pb-4 pt-2">
        <div className="max-w-2xl mx-auto">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about any electronics product..."
              disabled={isLoading}
              className="flex-1 px-4 py-3 rounded-xl text-[13px] outline-none transition-all duration-200 placeholder:text-[#444]"
              style={{
                background: '#141414',
                border: '1px solid #222',
                color: '#e8e8e8',
              }}
              onFocus={(e) => {
                e.currentTarget.style.borderColor = '#00d4aa';
                e.currentTarget.style.boxShadow = '0 0 0 2px rgba(0,212,170,0.1)';
              }}
              onBlur={(e) => {
                e.currentTarget.style.borderColor = '#222';
                e.currentTarget.style.boxShadow = 'none';
              }}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="w-11 h-11 rounded-xl flex items-center justify-center transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed"
              style={{
                background: input.trim() && !isLoading ? '#00d4aa' : '#1e1e1e',
              }}
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" style={{ color: '#555' }} />
              ) : (
                <Send
                  className="w-4 h-4"
                  style={{ color: input.trim() ? '#0a0a0a' : '#555' }}
                />
              )}
            </button>
          </form>

          <p className="text-center text-[10px] mt-3" style={{ color: '#333' }}>
            Powered by LangGraph RAG &middot; Built with LangChain + Supabase
          </p>
        </div>
      </div>
    </div>
  );
}
