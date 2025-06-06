export interface Message {
  id: number;
  type: 'user' | 'assistant' | 'error';
  content: string;
  timestamp: string;
}