"use client";

import { Bot } from 'lucide-react';

export function EmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center p-6">
      <div className="text-center py-12 animate-fade-in">
        <div className="relative mb-4 mx-auto">
          <div className="w-16 h-16 bg-gray-200 dark:bg-gray-700 rounded-full flex items-center justify-center mx-auto">
            <Bot className="w-8 h-8 text-gray-500 dark:text-gray-400" />
          </div>
          <div className="absolute -bottom-1 -right-1 w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center animate-pulse">
            <span className="w-4 h-4 bg-white rounded-full"></span>
          </div>
        </div>
        <h3 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-2">Start a conversation</h3>
        <p className="text-gray-500 dark:text-gray-400 max-w-sm mx-auto">
          Use the microphone to speak or type your message below to interact with the Qwen Omni Assistant
        </p>
      </div>
    </div>
  );
}