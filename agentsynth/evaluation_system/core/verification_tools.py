#!/usr/bin/env python3
"""
Additional Verification Tools for AgentSynth Task Evaluation

This module provides specialized tools for programmatic verification of computer use tasks
that go beyond basic PyAutoGUI functionality.
"""

import os
import subprocess
import json
import time
import re
import sqlite3
import psutil
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import numpy as np
import pytesseract
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class AdvancedVerificationTools:
    """
    Advanced verification tools for comprehensive task evaluation.
    """
    
    def __init__(self):
        self.temp_dir = Path("/tmp/agentsynth_verification")
        self.temp_dir.mkdir(exist_ok=True)
    
    # 1. SCREENSHOT ANALYSIS TOOLS
    def analyze_screenshot_for_elements(
        self, 
        screenshot_path: str, 
        expected_elements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze screenshot for specific UI elements using computer vision.
        
        Args:
            screenshot_path: Path to screenshot image
            expected_elements: List of expected UI elements with their properties
            
        Returns:
            Analysis results with found/missing elements
        """
        
        results = {
            'elements_found': [],
            'elements_missing': [],
            'confidence_scores': {},
            'overall_success': True
        }
        
        try:
            # Load image
            image = cv2.imread(screenshot_path)
            if image is None:
                return {'error': 'Could not load screenshot', 'overall_success': False}
            
            # Convert to different color spaces for better detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            for element in expected_elements:
                element_type = element.get('type', 'text')
                element_text = element.get('text', '')
                element_color = element.get('color', None)
                element_position = element.get('position', None)
                
                found = False
                confidence = 0.0
                
                if element_type == 'text':
                    found, confidence = self._detect_text_in_image(image, element_text)
                elif element_type == 'button':
                    found, confidence = self._detect_button_in_image(image, element)
                elif element_type == 'color_region':
                    found, confidence = self._detect_color_region(image, element_color)
                elif element_type == 'icon':
                    found, confidence = self._detect_icon_in_image(image, element)
                
                if found:
                    results['elements_found'].append(element)
                    results['confidence_scores'][element_text or element_type] = confidence
                else:
                    results['elements_missing'].append(element)
                    results['overall_success'] = False
        
        except Exception as e:
            results['error'] = str(e)
            results['overall_success'] = False
        
        return results
    
    def _detect_text_in_image(self, image: np.ndarray, text: str) -> Tuple[bool, float]:
        """Detect specific text in image using OCR."""
        try:
            # Use pytesseract for OCR
            ocr_text = pytesseract.image_to_string(image)
            text_found = text.lower() in ocr_text.lower()
            confidence = 1.0 if text_found else 0.0
            return text_found, confidence
        except:
            return False, 0.0
    
    def _detect_button_in_image(self, image: np.ndarray, button_info: Dict[str, Any]) -> Tuple[bool, float]:
        """Detect button-like elements in image."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for rectangular contours (potential buttons)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Check if it looks like a button (rectangular, reasonable size)
                if 0.5 < aspect_ratio < 3.0 and w > 50 and h > 20:
                    # Extract region and check for text
                    roi = image[y:y+h, x:x+w]
                    roi_text = pytesseract.image_to_string(roi)
                    
                    if button_info.get('text', '').lower() in roi_text.lower():
                        return True, 0.8
            
            return False, 0.0
        except:
            return False, 0.0
    
    def _detect_color_region(self, image: np.ndarray, target_color: str) -> Tuple[bool, float]:
        """Detect regions with specific colors."""
        try:
            # Convert color name to HSV range
            color_ranges = {
                'red': ([0, 50, 50], [10, 255, 255]),
                'green': ([40, 50, 50], [80, 255, 255]),
                'blue': ([100, 50, 50], [130, 255, 255]),
                'yellow': ([20, 50, 50], [40, 255, 255])
            }
            
            if target_color not in color_ranges:
                return False, 0.0
            
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower, upper = color_ranges[target_color]
            
            # Create mask
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Count pixels
            pixel_count = cv2.countNonZero(mask)
            total_pixels = image.shape[0] * image.shape[1]
            color_ratio = pixel_count / total_pixels
            
            # Consider it found if color covers more than 1% of image
            found = color_ratio > 0.01
            confidence = min(1.0, color_ratio * 10)
            
            return found, confidence
        except:
            return False, 0.0
    
    def _detect_icon_in_image(self, image: np.ndarray, icon_info: Dict[str, Any]) -> Tuple[bool, float]:
        """Detect specific icons in image using template matching."""
        try:
            # This would require icon templates - simplified implementation
            # In practice, you'd load icon templates and use cv2.matchTemplate
            return False, 0.0
        except:
            return False, 0.0
    
    # 2. WEB AUTOMATION VERIFICATION
    def verify_web_page_state(
        self, 
        url: str, 
        expected_elements: List[Dict[str, Any]],
        browser_type: str = 'chrome'
    ) -> Dict[str, Any]:
        """
        Verify web page state using Selenium WebDriver.
        
        Args:
            url: URL to check
            expected_elements: List of expected page elements
            browser_type: Browser to use ('chrome', 'firefox')
            
        Returns:
            Verification results
        """
        
        results = {
            'url_accessible': False,
            'elements_found': [],
            'elements_missing': [],
            'page_title': '',
            'page_source_length': 0,
            'overall_success': True
        }
        
        driver = None
        try:
            # Setup WebDriver
            if browser_type == 'chrome':
                options = webdriver.ChromeOptions()
                options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                driver = webdriver.Chrome(options=options)
            elif browser_type == 'firefox':
                options = webdriver.FirefoxOptions()
                options.add_argument('--headless')
                driver = webdriver.Firefox(options=options)
            
            # Navigate to URL
            driver.get(url)
            results['url_accessible'] = True
            results['page_title'] = driver.title
            results['page_source_length'] = len(driver.page_source)
            
            # Check for expected elements
            for element in expected_elements:
                element_type = element.get('type', 'id')
                element_value = element.get('value', '')
                element_text = element.get('text', '')
                
                found = False
                
                try:
                    if element_type == 'id':
                        elem = driver.find_element(By.ID, element_value)
                        found = True
                    elif element_type == 'class':
                        elem = driver.find_element(By.CLASS_NAME, element_value)
                        found = True
                    elif element_type == 'xpath':
                        elem = driver.find_element(By.XPATH, element_value)
                        found = True
                    elif element_type == 'text':
                        elem = driver.find_element(By.XPATH, f"//*[contains(text(), '{element_text}')]")
                        found = True
                    
                    if found:
                        results['elements_found'].append(element)
                    else:
                        results['elements_missing'].append(element)
                        results['overall_success'] = False
                        
                except Exception:
                    results['elements_missing'].append(element)
                    results['overall_success'] = False
        
        except Exception as e:
            results['error'] = str(e)
            results['overall_success'] = False
        
        finally:
            if driver:
                driver.quit()
        
        return results
    
    # 3. FILE SYSTEM VERIFICATION
    def verify_file_system_changes(
        self, 
        before_state: Dict[str, Any], 
        after_state: Dict[str, Any],
        expected_changes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify file system changes between two states.
        
        Args:
            before_state: File system state before task
            after_state: File system state after task
            expected_changes: List of expected changes
            
        Returns:
            Verification results
        """
        
        results = {
            'files_created': [],
            'files_modified': [],
            'files_deleted': [],
            'expected_changes_verified': [],
            'missing_changes': [],
            'overall_success': True
        }
        
        before_files = before_state.get('files', {})
        after_files = after_state.get('files', {})
        
        # Find created files
        for file_path in after_files:
            if file_path not in before_files:
                results['files_created'].append(file_path)
        
        # Find modified files
        for file_path in after_files:
            if file_path in before_files:
                before_mtime = before_files[file_path].get('modified', 0)
                after_mtime = after_files[file_path].get('modified', 0)
                if after_mtime > before_mtime:
                    results['files_modified'].append(file_path)
        
        # Find deleted files
        for file_path in before_files:
            if file_path not in after_files:
                results['files_deleted'].append(file_path)
        
        # Verify expected changes
        for expected_change in expected_changes:
            change_type = expected_change.get('type', 'create')
            file_path = expected_change.get('file_path', '')
            
            verified = False
            
            if change_type == 'create' and file_path in results['files_created']:
                verified = True
            elif change_type == 'modify' and file_path in results['files_modified']:
                verified = True
            elif change_type == 'delete' and file_path in results['files_deleted']:
                verified = True
            
            if verified:
                results['expected_changes_verified'].append(expected_change)
            else:
                results['missing_changes'].append(expected_change)
                results['overall_success'] = False
        
        return results
    
    # 4. PROCESS AND SYSTEM VERIFICATION
    def verify_process_changes(
        self, 
        before_processes: List[Dict[str, Any]], 
        after_processes: List[Dict[str, Any]],
        expected_processes: List[str]
    ) -> Dict[str, Any]:
        """
        Verify process changes between two states.
        
        Args:
            before_processes: Process list before task
            after_processes: Process list after task
            expected_processes: List of expected process names
            
        Returns:
            Verification results
        """
        
        results = {
            'processes_started': [],
            'processes_stopped': [],
            'expected_processes_running': [],
            'missing_processes': [],
            'overall_success': True
        }
        
        before_names = {proc.get('name', '') for proc in before_processes}
        after_names = {proc.get('name', '') for proc in after_processes}
        
        # Find started processes
        results['processes_started'] = list(after_names - before_names)
        
        # Find stopped processes
        results['processes_stopped'] = list(before_names - after_names)
        
        # Check expected processes
        for expected_process in expected_processes:
            if expected_process in after_names:
                results['expected_processes_running'].append(expected_process)
            else:
                results['missing_processes'].append(expected_process)
                results['overall_success'] = False
        
        return results
    
    # 5. DATABASE VERIFICATION
    def verify_database_state(
        self, 
        db_path: str, 
        expected_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify database state against expected conditions.
        
        Args:
            db_path: Path to database file
            expected_state: Expected database state
            
        Returns:
            Verification results
        """
        
        results = {
            'database_accessible': False,
            'tables_exist': [],
            'tables_missing': [],
            'row_counts': {},
            'data_verification': {},
            'overall_success': True
        }
        
        if not os.path.exists(db_path):
            results['error'] = 'Database file not found'
            return results
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            results['database_accessible'] = True
            
            # Check table existence
            expected_tables = expected_state.get('tables', [])
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            for table in expected_tables:
                if table in existing_tables:
                    results['tables_exist'].append(table)
                else:
                    results['tables_missing'].append(table)
                    results['overall_success'] = False
            
            # Check row counts
            expected_counts = expected_state.get('row_counts', {})
            for table, expected_count in expected_counts.items():
                if table in existing_tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    actual_count = cursor.fetchone()[0]
                    results['row_counts'][table] = actual_count
                    
                    if actual_count != expected_count:
                        results['overall_success'] = False
            
            # Verify specific data
            data_checks = expected_state.get('data_checks', [])
            for check in data_checks:
                table = check['table']
                query = check['query']
                expected_result = check['expected_result']
                
                if table in existing_tables:
                    cursor.execute(query)
                    actual_result = cursor.fetchall()
                    
                    if actual_result == expected_result:
                        results['data_verification'][f"{table}_{query[:20]}"] = 'PASS'
                    else:
                        results['data_verification'][f"{table}_{query[:20]}"] = 'FAIL'
                        results['overall_success'] = False
            
            conn.close()
            
        except Exception as e:
            results['error'] = str(e)
            results['overall_success'] = False
        
        return results
    
    # 6. NETWORK AND API VERIFICATION
    def verify_network_state(
        self, 
        expected_connections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify network connections and API responses.
        
        Args:
            expected_connections: List of expected network connections
            
        Returns:
            Verification results
        """
        
        results = {
            'connections_successful': [],
            'connections_failed': [],
            'api_responses': {},
            'overall_success': True
        }
        
        for connection in expected_connections:
            connection_type = connection.get('type', 'http')
            url = connection.get('url', '')
            expected_status = connection.get('expected_status', 200)
            expected_content = connection.get('expected_content', '')
            
            try:
                if connection_type == 'http':
                    response = requests.get(url, timeout=10)
                    results['api_responses'][url] = {
                        'status_code': response.status_code,
                        'content_length': len(response.text)
                    }
                    
                    if response.status_code == expected_status:
                        results['connections_successful'].append(connection)
                        
                        if expected_content and expected_content in response.text:
                            results['api_responses'][url]['content_found'] = True
                        else:
                            results['api_responses'][url]['content_found'] = False
                            results['overall_success'] = False
                    else:
                        results['connections_failed'].append(connection)
                        results['overall_success'] = False
                        
            except Exception as e:
                results['connections_failed'].append(connection)
                results['api_responses'][url] = {'error': str(e)}
                results['overall_success'] = False
        
        return results

# Example usage and integration
def create_comprehensive_verification_example():
    """Create a comprehensive example of verifiable evaluation."""
    
    tools = AdvancedVerificationTools()
    
    # Example: Verify a web form submission task
    web_verification = tools.verify_web_page_state(
        url="https://httpbin.org/forms/post",
        expected_elements=[
            {'type': 'text', 'text': 'Customer name'},
            {'type': 'text', 'text': 'Email'},
            {'type': 'id', 'value': 'submitbutton'}
        ]
    )
    
    # Example: Verify file creation
    before_state = {'files': {}}
    after_state = {
        'files': {
            '/tmp/test.txt': {'size': 12, 'modified': time.time()}
        }
    }
    
    file_verification = tools.verify_file_system_changes(
        before_state=before_state,
        after_state=after_state,
        expected_changes=[
            {'type': 'create', 'file_path': '/tmp/test.txt'}
        ]
    )
    
    return {
        'web_verification': web_verification,
        'file_verification': file_verification
    }

if __name__ == "__main__":
    # Run example
    example_results = create_comprehensive_verification_example()
    print("Comprehensive Verification Example:")
    print(json.dumps(example_results, indent=2))

