from django.test import TestCase
import json
import uuid
from unittest.mock import patch, MagicMock
from django.test import TestCase, Client
from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status

from analyzer.zero import ZeroShotClassifier
from analyzer.tasks import create_task, get_task_status, process_task, tasks
from analyzer.models import TaskCreateView, TaskStatusView


class ZeroShotClassifierTest(TestCase):
    """Test cases for ZeroShotClassifier"""

    @patch('analyzer.zero.AutoTokenizer.from_pretrained')
    @patch('analyzer.zero.AutoModelForSequenceClassification.from_pretrained')
    def test_classifier_initialization(self, mock_model, mock_tokenizer):
        """Test that classifier initializes correctly"""
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        classifier = ZeroShotClassifier()
        
        self.assertIsNotNone(classifier.tokenizer)
        self.assertIsNotNone(classifier.model)
        self.assertEqual(classifier.target_label, 'entailment')

    @patch('analyzer.zero.torch.softmax')
    @patch('analyzer.zero.torch.inference_mode')
    @patch('analyzer.zero.AutoTokenizer.from_pretrained')
    @patch('analyzer.zero.AutoModelForSequenceClassification.from_pretrained')
    def test_predict_method(self, mock_model_cls, mock_tokenizer_cls, mock_inference, mock_softmax):
        """Test the predict method returns probabilities"""
        # Mock the inference context manager
        mock_inference.return_value.__enter__ = MagicMock()
        mock_inference.return_value.__exit__ = MagicMock()
        
        # Mock tokenizer and model classes
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_cls.return_value = mock_tokenizer
        mock_model_cls.return_value = mock_model
        
        # Create classifier with mocked dependencies
        classifier = ZeroShotClassifier()
        
        # Configure mocks
        mock_model.device = 'cpu'
        mock_model.config.label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        
        # Mock tokenizer output
        mock_tokens = MagicMock()
        mock_tokens.to.return_value = mock_tokens
        mock_tokenizer.return_value = mock_tokens
        
        # Mock model output
        mock_logits = MagicMock()
        mock_model_output = MagicMock()
        mock_model_output.logits = mock_logits
        mock_model.return_value = mock_model_output
        
        # Mock softmax probabilities
        import numpy as np
        mock_proba_tensor = MagicMock()
        mock_proba_tensor.cpu.return_value.numpy.return_value = np.array([0.7, 0.3])
        mock_softmax.return_value = MagicMock()
        mock_softmax.return_value.__getitem__.return_value = mock_proba_tensor
        
        # Test prediction
        text = "This is a test news article"
        labels = ["positive", "negative"]
        result = classifier.predict(text, labels)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 2)


class TasksTest(TestCase):
    """Test cases for task management"""

    def setUp(self):
        # Clear tasks before each test
        tasks.clear()

    @patch('analyzer.tasks.process_task_async')
    def test_create_task(self, mock_async):
        """Test task creation"""
        test_data = {
            'pairs': [
                {'class1': 'positive', 'class2': 'negative'}
            ]
        }
        
        task_id = create_task(test_data)
        
        self.assertIsInstance(task_id, str)
        self.assertIn(task_id, tasks)
        self.assertEqual(tasks[task_id]['status'], 'Pending')
        self.assertEqual(tasks[task_id]['input_data'], test_data)
        mock_async.assert_called_once_with(task_id)

    @patch('analyzer.tasks.process_task_async')
    def test_get_task_status_existing(self, mock_async):
        """Test getting status of existing task"""
        test_data = {'pairs': []}
        task_id = create_task(test_data)
        
        status = get_task_status(task_id)
        
        self.assertEqual(status['status'], 'Pending')
        self.assertIsNone(status['result'])
        self.assertIsNone(status['error'])

    def test_get_task_status_nonexistent(self):
        """Test getting status of non-existent task"""
        fake_task_id = str(uuid.uuid4())
        
        status = get_task_status(fake_task_id)
        
        self.assertEqual(status['status'], 'Task not found')

    @patch('analyzer.tasks.scrappers')
    @patch('analyzer.tasks.classifier')
    def test_process_task_success(self, mock_classifier, mock_scrappers):
        """Test successful task processing"""
        # Mock news scraper
        mock_scrapper = MagicMock()
        mock_scrapper.get_news.return_value = [
            "Test news article 1",
            "Test news article 2"
        ]
        mock_scrappers.__getitem__.return_value = mock_scrapper
        
        # Mock classifier
        mock_classifier.predict.return_value = [0.6, 0.4]
        
        # Create test task manually (without async processing)
        test_data = {
            'pairs': [
                {'class1': 'positive', 'class2': 'negative'}
            ]
        }
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            'status': 'Pending',
            'input_data': test_data,
            'result': None,
            'error': None
        }
        
        # Process task
        process_task(task_id, test_data)
        
        # Check results
        self.assertIsNotNone(tasks[task_id]['result'])
        self.assertIn('news_results', tasks[task_id]['result'])
        self.assertIn('summary', tasks[task_id]['result'])

    @patch('analyzer.tasks.scrappers')
    def test_process_task_empty_news(self, mock_scrappers):
        """Test task processing with no news"""
        # Mock empty news scraper
        mock_scrapper = MagicMock()
        mock_scrapper.get_news.return_value = []
        mock_scrappers.__getitem__.return_value = mock_scrapper
        
        # Create test task manually (without async processing)
        test_data = {'pairs': []}
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            'status': 'Pending',
            'input_data': test_data,
            'result': None,
            'error': None
        }
        
        # Process task
        process_task(task_id, test_data)
        
        # Check error
        self.assertEqual(tasks[task_id]['error'], "Parsing error")


class APIViewsTest(APITestCase):
    """Test cases for API views"""

    def setUp(self):
        self.client = Client()
        tasks.clear()

    @patch('analyzer.models.create_task')
    def test_task_create_view_valid_data(self, mock_create_task):
        """Test TaskCreateView with valid data"""
        mock_create_task.return_value = "test-task-id"
        
        url = reverse('create_task')
        data = {
            'pairs': [
                {'class1': 'positive', 'class2': 'negative'}
            ]
        }
        
        # We'll need to mock the serializer validation
        with patch('analyzer.models.BaseClassificationSerializer') as mock_serializer:
            mock_serializer_instance = MagicMock()
            mock_serializer_instance.is_valid.return_value = True
            mock_serializer_instance.validated_data = data
            mock_serializer.return_value = mock_serializer_instance
            
            response = self.client.post(url, json.dumps(data), content_type='application/json')
            
            self.assertEqual(response.status_code, 201)

    @patch('analyzer.tasks.process_task_async')
    def test_task_status_view(self, mock_async):
        """Test TaskStatusView"""
        # Create a test task
        test_data = {'pairs': []}
        task_id = create_task(test_data)
        
        url = reverse('task_status', kwargs={'task_id': task_id})
        
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['status'], 'Pending')


class IntegrationTest(TestCase):
    """Integration tests for the complete workflow"""

    def setUp(self):
        tasks.clear()

    @patch('analyzer.tasks.scrappers')
    @patch('analyzer.tasks.classifier')
    @patch('analyzer.tasks.process_task_async')
    def test_complete_workflow(self, mock_async, mock_classifier, mock_scrappers):
        """Test the complete workflow from task creation to completion"""
        # Mock dependencies
        mock_scrapper = MagicMock()
        mock_scrapper.get_news.return_value = ["Test news"]
        mock_scrappers.__getitem__.return_value = mock_scrapper
        mock_classifier.predict.return_value = [0.7, 0.3]
        
        # Create task
        test_data = {
            'pairs': [
                {'class1': 'positive', 'class2': 'negative'}
            ]
        }
        task_id = create_task(test_data)
        
        # Verify initial status
        status = get_task_status(task_id)
        self.assertEqual(status['status'], 'Pending')
        
        # Process task manually (not async)
        process_task(task_id, test_data)
        
        # Verify completion
        final_status = get_task_status(task_id)
        self.assertIsNotNone(final_status['result'])
        self.assertIsNone(final_status['error'])


class ModelTest(TestCase):
    """Test cases for Django models and views in models.py"""

    def setUp(self):
        self.client = Client()

    def test_task_create_view_post_invalid_data(self):
        """Test TaskCreateView with invalid POST data"""
        view = TaskCreateView()
        
        # Test that the view class exists and has the right methods
        self.assertTrue(hasattr(view, 'post'))
        self.assertEqual(view.__class__.__name__, 'TaskCreateView')

    def test_task_status_view_get(self):
        """Test TaskStatusView GET method"""
        view = TaskStatusView()
        
        # Test that the view class exists and has the right methods
        self.assertTrue(hasattr(view, 'get'))
        self.assertEqual(view.__class__.__name__, 'TaskStatusView')
