import json
import logging
from datetime import datetime
from constants import LOG_FILE_PATH
import os
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RadioProtectLogger')

def write_log(log_entry):
    try:
        with open(LOG_FILE_PATH, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        logger.error(f"Error writing to log file: {e}")

def log_treatment_check(technician_id, patient_id, check_outcome, additional_info=None):
    log_entry = {
        'log_id': str(uuid.uuid4()),
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'event': 'treatment_check performed',
        'technician_id': technician_id,
        'patient_id': patient_id,
        'safety_check_outcome': check_outcome,
        'additional_info': additional_info or 'N/A'
    }
    write_log(log_entry)

def log_doctor_review(doctor_id, patient_id, technician_id, review_outcome, additional_info=None):
    log_entry = {
        'log_id': str(uuid.uuid4()),
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'event': 'doctor review performed',
        'doctor_id': doctor_id,
        'patient_id': patient_id,
        'technician_id': technician_id,
        'review_outcome': review_outcome,
        'additional_info': additional_info or 'N/A'
    }
    write_log(log_entry)

def replanning_needed(technician_id, patient_id, additional_info=None):
    log_entry = {
        'log_id': str(uuid.uuid4()),
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'event': 'dangerous contours detected, replanning needed',
        'technician_id': technician_id,
        'patient_id': patient_id,
        'additional_info': additional_info or 'N/A'
    }
    write_log(log_entry)

def log_treatment_proceeded(technician_id, patient_id, check_outcome, additional_info=None):
    log_entry = {
        'log_id': str(uuid.uuid4()),
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'event': 'treatment proceeded',
        'technician_id': technician_id,
        'patient_id': patient_id,
        'previous_safety_check_outcome': check_outcome,
        'additional_info': additional_info or 'N/A'
    }
    write_log(log_entry)
