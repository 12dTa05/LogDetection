import re
from typing import Optional

def extract_block_id(log_message: str) -> Optional[str]:
    match = re.search(r'blk_[-]?\d+', log_message)
    if match:
        return match.group()
    return None

def clean_log_message(log_message: str) -> str:
    # Remove IP addresses
    cleaned = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?', '<IP>', log_message)
    # Remove paths
    cleaned = re.sub(r'/[\w/]+', '<PATH>', cleaned)
    return cleaned

def get_parameter_list(message: str, template: str) -> list:
    # Split template and message into tokens
    template_tokens = template.split()
    message_tokens = message.split()
    
    params = []
    for t_token, m_token in zip(template_tokens, message_tokens):
        if t_token == '<*>':
            params.append(m_token)
    
    return params
