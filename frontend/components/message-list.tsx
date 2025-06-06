"use client";

import { useEffect, useRef } from 'react';
import { Bot, User } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Message } from '@/types/chat';
import { ScrollArea } from '@/components/ui/scroll-area';
import { EmptyState } from '@/components/empty-state';

interface MessageListProps {
  messages: Message[];
}

export function MessageList({ messages }: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return <EmptyState />;
  }

  return (
    <ScrollArea className="flex-1 p-6">
      <div className="space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={cn(
              "flex animate-fade-in",
              message.type === 'user' ? "justify-end" : "justify-start"
            )}
          >
            <div className={cn(
              "flex items-start space-x-3 max-w-3xl",
              message.type === 'user' ? "flex-row-reverse space-x-reverse" : ""
            )}>
              <div className={cn(
                "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0",
                message.type === 'user' 
                  ? "bg-blue-500" 
                  : message.type === 'error'
                  ? "bg-red-500"
                  : "bg-gradient-to-r from-purple-500 to-indigo-600"
              )}>
                {message.type === 'user' ? (
                  <User className="w-4 h-4 text-white" />
                ) : (
                  <Bot className="w-4 h-4 text-white" />
                )}
              </div>
              
              <div className={cn(
                "px-4 py-3 rounded-2xl",
                message.type === 'user'
                  ? "bg-blue-500 text-white"
                  : message.type === 'error'
                  ? "bg-red-100 text-red-800 border border-red-200 dark:bg-red-900/30 dark:text-red-300 dark:border-red-800/50"
                  : "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200"
              )}>
                <p className="whitespace-pre-wrap">{message.content}</p>
                <p className={cn(
                  "text-xs mt-1 opacity-70",
                  message.type === 'user' 
                    ? "text-blue-100" 
                    : message.type === 'error'
                    ? "text-red-400"
                    : "text-gray-500 dark:text-gray-400"
                )}>
                  {new Date(message.timestamp).toLocaleTimeString()}
                </p>
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
    </ScrollArea>
  );
}