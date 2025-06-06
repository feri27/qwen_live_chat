"use client";

import { useEffect } from 'react';
import { Bot } from 'lucide-react';
import { useWebSocket } from '@/hooks/use-websocket';
import { useVoiceRecording } from '@/hooks/use-voice-recording';
import { ChatHeader } from '@/components/chat-header';
import { MessageList } from '@/components/message-list';
import { InputArea } from '@/components/input-area';

export function QwenOmniChat() {
  const {
    isConnected,
    status,
    messages,
    clientId,
    sendTextMessage,
    clearConversation
  } = useWebSocket();

  const {
    isRecording,
    toggleRecording,
    startRecording,
    stopRecording
  } = useVoiceRecording({ wsRef: useWebSocket().wsRef, isConnected });

  useEffect(() => {
    // This ensures recording stops if the component unmounts
    return () => {
      if (isRecording) {
        stopRecording();
      }
    };
  }, [isRecording, stopRecording]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-indigo-950 flex flex-col">
      <ChatHeader 
        isConnected={isConnected} 
        status={status} 
        clearConversation={clearConversation} 
      />

      <div className="flex-1 max-w-4xl mx-auto w-full px-4 py-6 overflow-hidden">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg h-full flex flex-col">
          <MessageList messages={messages} />

          <InputArea
            isConnected={isConnected}
            isRecording={isRecording}
            toggleRecording={toggleRecording}
            sendTextMessage={sendTextMessage}
          />
        </div>
      </div>
    </div>
  );
}