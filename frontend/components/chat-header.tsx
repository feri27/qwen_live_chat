"use client";

import { Bot, Trash2 } from 'lucide-react';
import { 
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface ChatHeaderProps {
  isConnected: boolean;
  status: string;
  clearConversation: () => Promise<void>;
}

export function ChatHeader({ isConnected, status, clearConversation }: ChatHeaderProps) {
  return (
    <div className="bg-white dark:bg-gray-800 shadow-lg border-b dark:border-gray-700">
      <div className="max-w-4xl mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full flex items-center justify-center">
              <Bot className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-800 dark:text-gray-100">Qwen Omni Assistant</h1>
              <p className="text-sm text-gray-500 dark:text-gray-400">AI-Powered Voice & Text Chat</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className={cn(
              "px-3 py-1 rounded-full text-sm font-medium transition-colors",
              isConnected 
                ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400" 
                : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"
            )}>
              {status}
            </div>
            
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    onClick={clearConversation}
                    variant="ghost"
                    size="icon"
                    className="text-gray-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-full"
                  >
                    <Trash2 className="w-5 h-5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Clear conversation</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        </div>
      </div>
    </div>
  );
}