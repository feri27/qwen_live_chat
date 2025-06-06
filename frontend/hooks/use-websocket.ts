"use client";

import { useState, useRef, useCallback, useEffect } from 'react';
import { Message } from '@/types/chat';

// Get the WebSocket URL from environment or construct it from the current window location
const getWebSocketUrl = () => {
  if (typeof window === 'undefined') return '';
  
  // If NEXT_PUBLIC_WEBSOCKET_URL is set, use it
  if (process.env.NEXT_PUBLIC_WEBSOCKET_URL) {
    return process.env.NEXT_PUBLIC_WEBSOCKET_URL;
  }

  // Otherwise, construct the WebSocket URL from the current window location
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  return `${protocol}//${host}`;
};

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState('Disconnected');
  const [clientId] = useState(() => Math.random().toString(36).substr(2, 9));
  
  const wsRef = useRef<WebSocket | null>(null);

  // Initialize WebSocket connection
  const connectWebSocket = useCallback(() => {
    try {
      const wsUrl = getWebSocketUrl();
      if (!wsUrl) return;

      const ws = new WebSocket(`${wsUrl}/ws/${clientId}`);
      
      ws.onopen = () => {
        setIsConnected(true);
        setStatus('Connected');
        console.log('WebSocket connected');
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'status':
            setStatus(data.message);
            break;
          case 'transcription':
            setMessages(prev => [...prev, {
              id: Date.now(),
              type: 'user',
              content: data.text,
              timestamp: data.timestamp
            }]);
            setStatus('Generating response...');
            break;
          case 'response':
            setMessages(prev => [...prev, {
              id: Date.now() + 1,
              type: 'assistant',
              content: data.text,
              timestamp: data.timestamp
            }]);
            setStatus('Ready');
            break;
          case 'error':
            setStatus(`Error: ${data.message}`);
            setMessages(prev => [...prev, {
              id: Date.now(),
              type: 'error',
              content: data.message,
              timestamp: data.timestamp
            }]);
            break;
          default:
            console.log('Unknown message type:', data.type);
        }
      };
      
      ws.onclose = () => {
        setIsConnected(false);
        setStatus('Disconnected');
        console.log('WebSocket disconnected');
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setStatus('Connection error');
      };
      
      wsRef.current = ws;
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      setStatus('Connection failed');
    }
  }, [clientId]);

  // Initialize WebSocket on component mount
  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);

  // Send text message
  const sendTextMessage = useCallback((text: string) => {
    if (text.trim() && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'text',
        text: text.trim()
      }));
      
      setMessages(prev => [...prev, {
        id: Date.now(),
        type: 'user',
        content: text.trim(),
        timestamp: new Date().toISOString()
      }]);
      
      setStatus('Generating response...');
    }
  }, []);

  // Clear conversation
  const clearConversation = useCallback(async () => {
    const wsUrl = getWebSocketUrl();
    if (!wsUrl) return;

    try {
      await fetch(`${wsUrl.replace('ws:', 'http:').replace('wss:', 'https:')}/conversation/${clientId}`, {
        method: 'DELETE'
      });
      setMessages([]);
      setStatus('Conversation cleared');
    } catch (error) {
      console.error('Failed to clear conversation:', error);
    }
  }, [clientId]);

  return {
    isConnected,
    status,
    messages,
    clientId,
    wsRef,
    sendTextMessage,
    clearConversation
  };
}