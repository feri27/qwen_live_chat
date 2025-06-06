"use client";

import { useState, KeyboardEvent } from 'react';
import { Mic, MicOff, Send } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { cn } from '@/lib/utils';

interface InputAreaProps {
  isConnected: boolean;
  isRecording: boolean;
  toggleRecording: () => void;
  sendTextMessage: (text: string) => void;
}

export function InputArea({ 
  isConnected, 
  isRecording, 
  toggleRecording, 
  sendTextMessage 
}: InputAreaProps) {
  const [inputText, setInputText] = useState('');

  const handleSendMessage = () => {
    if (inputText.trim() && isConnected) {
      sendTextMessage(inputText.trim());
      setInputText('');
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="border-t p-4 dark:border-gray-700">
      <div className="flex items-center space-x-3">
        <Button
          onClick={toggleRecording}
          disabled={!isConnected}
          variant={isRecording ? "destructive" : "default"}
          size="icon"
          className={cn(
            "rounded-full h-12 w-12 flex-shrink-0",
            isRecording && "animate-pulse shadow-lg"
          )}
        >
          {isRecording ? (
            <MicOff className="w-5 h-5" />
          ) : (
            <Mic className="w-5 h-5" />
          )}
        </Button>
        
        <Textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="Type your message or use the microphone..."
          className="resize-none min-h-[48px] max-h-[120px]"
          disabled={!isConnected}
        />
        
        <Button
          onClick={handleSendMessage}
          disabled={!isConnected || !inputText.trim()}
          size="icon"
          variant="default"
          className="rounded-full h-12 w-12 flex-shrink-0 bg-green-500 hover:bg-green-600 text-white disabled:bg-gray-300"
        >
          <Send className="w-5 h-5" />
        </Button>
      </div>
    </div>
  );
}