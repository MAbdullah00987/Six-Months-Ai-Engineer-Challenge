
"""
Comprehensive Unit Tests for Stack and Queue
Professional-grade testing with edge cases and error handling

Run with: python test_stack_queue.py
"""

import unittest
from stack_and_queue import Stack, Queue, EmptyStackError, EmptyQueueError


class TestStack(unittest.TestCase):
    """Comprehensive test suite for Stack class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.stack = Stack()
        self.populated_stack = Stack([1, 2, 3, 4, 5])
    
    def test_init_empty(self):
        """Test creating an empty stack."""
        stack = Stack()
        self.assertTrue(stack.is_empty())
        self.assertEqual(len(stack), 0)
    
    def test_init_with_items(self):
        """Test creating a stack with initial items."""
        stack = Stack([1, 2, 3])
        self.assertEqual(len(stack), 3)
        self.assertEqual(stack.peek(), 3)
    
    def test_init_with_invalid_type(self):
        """Test that TypeError is raised for invalid initialization."""
        with self.assertRaises(TypeError):
            Stack("invalid")
    
    def test_push_single_item(self):
        """Test pushing a single item to the stack."""
        self.stack.push(10)
        self.assertEqual(len(self.stack), 1)
        self.assertEqual(self.stack.peek(), 10)
    
    def test_push_multiple_items(self):
        """Test pushing multiple items to the stack."""
        items = [1, 2, 3, 4, 5]
        for item in items:
            self.stack.push(item)
        self.assertEqual(len(self.stack), 5)
        self.assertEqual(self.stack.peek(), 5)
    
    def test_push_various_types(self):
        """Test pushing different data types."""
        self.stack.push(1)
        self.stack.push("string")
        self.stack.push([1, 2, 3])
        self.stack.push({'key': 'value'})
        self.assertEqual(len(self.stack), 4)
        self.assertEqual(self.stack.peek(), {'key': 'value'})
    
    def test_pop_single_item(self):
        """Test popping a single item from the stack."""
        self.stack.push(10)
        item = self.stack.pop()
        self.assertEqual(item, 10)
        self.assertTrue(self.stack.is_empty())
    
    def test_pop_multiple_items(self):
        """Test popping multiple items from the stack."""
        expected = [5, 4, 3, 2, 1]
        for expected_item in expected:
            item = self.populated_stack.pop()
            self.assertEqual(item, expected_item)
        self.assertTrue(self.populated_stack.is_empty())
    
    def test_pop_empty_stack(self):
        """Test that popping from empty stack raises EmptyStackError."""
        with self.assertRaises(EmptyStackError) as context:
            self.stack.pop()
        self.assertIn("empty stack", str(context.exception).lower())
    
    def test_peek(self):
        """Test peeking at the top item without removing it."""
        self.stack.push(10)
        self.stack.push(20)
        top = self.stack.peek()
        self.assertEqual(top, 20)
        self.assertEqual(len(self.stack), 2)  # Size unchanged
    
    def test_peek_empty_stack(self):
        """Test that peeking at empty stack raises EmptyStackError."""
        with self.assertRaises(EmptyStackError) as context:
            self.stack.peek()
        self.assertIn("empty stack", str(context.exception).lower())
    
    def test_is_empty_true(self):
        """Test is_empty returns True for empty stack."""
        self.assertTrue(self.stack.is_empty())
    
    def test_is_empty_false(self):
        """Test is_empty returns False for non-empty stack."""
        self.stack.push(1)
        self.assertFalse(self.stack.is_empty())
    
    def test_is_empty_after_operations(self):
        """Test is_empty after push and pop operations."""
        self.stack.push(1)
        self.assertFalse(self.stack.is_empty())
        self.stack.pop()
        self.assertTrue(self.stack.is_empty())
    
    def test_size(self):
        """Test size method returns correct count."""
        self.assertEqual(self.stack.size(), 0)
        self.stack.push(1)
        self.assertEqual(self.stack.size(), 1)
        self.stack.push(2)
        self.assertEqual(self.stack.size(), 2)
    
    def test_len_magic_method(self):
        """Test __len__ special method."""
        self.assertEqual(len(self.stack), 0)
        self.stack.push(1)
        self.assertEqual(len(self.stack), 1)
        self.assertEqual(len(self.populated_stack), 5)
    
    def test_clear(self):
        """Test clearing all items from the stack."""
        self.populated_stack.clear()
        self.assertTrue(self.populated_stack.is_empty())
        self.assertEqual(len(self.populated_stack), 0)
    
    def test_to_list(self):
        """Test converting stack to list."""
        result = self.populated_stack.to_list()
        self.assertEqual(result, [5, 4, 3, 2, 1])
    
    def test_str_empty(self):
        """Test string representation of empty stack."""
        self.assertIn("empty", str(self.stack).lower())
    
    def test_str_populated(self):
        """Test string representation of populated stack."""
        string_repr = str(self.populated_stack)
        self.assertIn("Stack", string_repr)
        self.assertIn("5", string_repr)
    
    def test_repr(self):
        """Test repr method."""
        repr_str = repr(self.populated_stack)
        self.assertIn("Stack", repr_str)
        self.assertIn("[1, 2, 3, 4, 5]", repr_str)
    
    def test_iter(self):
        """Test iterating through stack (top to bottom)."""
        items = list(self.populated_stack)
        self.assertEqual(items, [5, 4, 3, 2, 1])
    
    def test_contains_true(self):
        """Test 'in' operator returns True for existing item."""
        self.assertTrue(3 in self.populated_stack)
    
    def test_contains_false(self):
        """Test 'in' operator returns False for non-existing item."""
        self.assertFalse(10 in self.populated_stack)
    
    def test_equality_same_stacks(self):
        """Test equality of two identical stacks."""
        stack1 = Stack([1, 2, 3])
        stack2 = Stack([1, 2, 3])
        self.assertEqual(stack1, stack2)
    
    def test_equality_different_stacks(self):
        """Test inequality of different stacks."""
        stack1 = Stack([1, 2, 3])
        stack2 = Stack([1, 2, 4])
        self.assertNotEqual(stack1, stack2)
    
    def test_equality_with_non_stack(self):
        """Test equality comparison with non-Stack object."""
        stack = Stack([1, 2, 3])
        self.assertNotEqual(stack, [1, 2, 3])
    
    def test_push_pop_sequence(self):
        """Test complex push-pop sequence."""
        self.stack.push(1)
        self.stack.push(2)
        self.assertEqual(self.stack.pop(), 2)
        self.stack.push(3)
        self.stack.push(4)
        self.assertEqual(self.stack.pop(), 4)
        self.assertEqual(self.stack.pop(), 3)
        self.assertEqual(self.stack.pop(), 1)
    
    def test_large_stack(self):
        """Test stack with large number of items."""
        large_stack = Stack()
        n = 10000
        for i in range(n):
            large_stack.push(i)
        self.assertEqual(len(large_stack), n)
        self.assertEqual(large_stack.pop(), n - 1)


class TestQueue(unittest.TestCase):
    """Comprehensive test suite for Queue class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.queue = Queue()
        self.populated_queue = Queue([1, 2, 3, 4, 5])
    
    def test_init_empty(self):
        """Test creating an empty queue."""
        queue = Queue()
        self.assertTrue(queue.is_empty())
        self.assertEqual(len(queue), 0)
    
    def test_init_with_items(self):
        """Test creating a queue with initial items."""
        queue = Queue([1, 2, 3])
        self.assertEqual(len(queue), 3)
        self.assertEqual(queue.peek(), 1)
    
    def test_init_with_invalid_type(self):
        """Test that TypeError is raised for invalid initialization."""
        with self.assertRaises(TypeError):
            Queue(123)
    
    def test_enqueue_single_item(self):
        """Test enqueueing a single item to the queue."""
        self.queue.enqueue(10)
        self.assertEqual(len(self.queue), 1)
        self.assertEqual(self.queue.peek(), 10)
    
    def test_enqueue_multiple_items(self):
        """Test enqueueing multiple items to the queue."""
        items = [1, 2, 3, 4, 5]
        for item in items:
            self.queue.enqueue(item)
        self.assertEqual(len(self.queue), 5)
        self.assertEqual(self.queue.peek(), 1)
    
    def test_enqueue_various_types(self):
        """Test enqueueing different data types."""
        self.queue.enqueue(1)
        self.queue.enqueue("string")
        self.queue.enqueue([1, 2, 3])
        self.queue.enqueue({'key': 'value'})
        self.assertEqual(len(self.queue), 4)
        self.assertEqual(self.queue.peek(), 1)
    
    def test_dequeue_single_item(self):
        """Test dequeueing a single item from the queue."""
        self.queue.enqueue(10)
        item = self.queue.dequeue()
        self.assertEqual(item, 10)
        self.assertTrue(self.queue.is_empty())
    
    def test_dequeue_multiple_items(self):
        """Test dequeueing multiple items from the queue."""
        expected = [1, 2, 3, 4, 5]
        for expected_item in expected:
            item = self.populated_queue.dequeue()
            self.assertEqual(item, expected_item)
        self.assertTrue(self.populated_queue.is_empty())
    
    def test_dequeue_empty_queue(self):
        """Test that dequeueing from empty queue raises EmptyQueueError."""
        with self.assertRaises(EmptyQueueError) as context:
            self.queue.dequeue()
        self.assertIn("empty queue", str(context.exception).lower())
    
    def test_peek(self):
        """Test peeking at the front item without removing it."""
        self.queue.enqueue(10)
        self.queue.enqueue(20)
        front = self.queue.peek()
        self.assertEqual(front, 10)
        self.assertEqual(len(self.queue), 2)  # Size unchanged
    
    def test_peek_empty_queue(self):
        """Test that peeking at empty queue raises EmptyQueueError."""
        with self.assertRaises(EmptyQueueError) as context:
            self.queue.peek()
        self.assertIn("empty queue", str(context.exception).lower())
    
    def test_is_empty_true(self):
        """Test is_empty returns True for empty queue."""
        self.assertTrue(self.queue.is_empty())
    
    def test_is_empty_false(self):
        """Test is_empty returns False for non-empty queue."""
        self.queue.enqueue(1)
        self.assertFalse(self.queue.is_empty())
    
    def test_is_empty_after_operations(self):
        """Test is_empty after enqueue and dequeue operations."""
        self.queue.enqueue(1)
        self.assertFalse(self.queue.is_empty())
        self.queue.dequeue()
        self.assertTrue(self.queue.is_empty())
    
    def test_size(self):
        """Test size method returns correct count."""
        self.assertEqual(self.queue.size(), 0)
        self.queue.enqueue(1)
        self.assertEqual(self.queue.size(), 1)
        self.queue.enqueue(2)
        self.assertEqual(self.queue.size(), 2)
    
    def test_len_magic_method(self):
        """Test __len__ special method."""
        self.assertEqual(len(self.queue), 0)
        self.queue.enqueue(1)
        self.assertEqual(len(self.queue), 1)
        self.assertEqual(len(self.populated_queue), 5)
    
    def test_clear(self):
        """Test clearing all items from the queue."""
        self.populated_queue.clear()
        self.assertTrue(self.populated_queue.is_empty())
        self.assertEqual(len(self.populated_queue), 0)
    
    def test_to_list(self):
        """Test converting queue to list."""
        result = self.populated_queue.to_list()
        self.assertEqual(result, [1, 2, 3, 4, 5])
    
    def test_str_empty(self):
        """Test string representation of empty queue."""
        self.assertIn("empty", str(self.queue).lower())
    
    def test_str_populated(self):
        """Test string representation of populated queue."""
        string_repr = str(self.populated_queue)
        self.assertIn("Queue", string_repr)
        self.assertIn("1", string_repr)
    
    def test_repr(self):
        """Test repr method."""
        repr_str = repr(self.populated_queue)
        self.assertIn("Queue", repr_str)
        self.assertIn("[1, 2, 3, 4, 5]", repr_str)
    
    def test_iter(self):
        """Test iterating through queue (front to rear)."""
        items = list(self.populated_queue)
        self.assertEqual(items, [1, 2, 3, 4, 5])
    
    def test_contains_true(self):
        """Test 'in' operator returns True for existing item."""
        self.assertTrue(3 in self.populated_queue)
    
    def test_contains_false(self):
        """Test 'in' operator returns False for non-existing item."""
        self.assertFalse(10 in self.populated_queue)
    
    def test_equality_same_queues(self):
        """Test equality of two identical queues."""
        queue1 = Queue([1, 2, 3])
        queue2 = Queue([1, 2, 3])
        self.assertEqual(queue1, queue2)
    
    def test_equality_different_queues(self):
        """Test inequality of different queues."""
        queue1 = Queue([1, 2, 3])
        queue2 = Queue([1, 2, 4])
        self.assertNotEqual(queue1, queue2)
    
    def test_equality_with_non_queue(self):
        """Test equality comparison with non-Queue object."""
        queue = Queue([1, 2, 3])
        self.assertNotEqual(queue, [1, 2, 3])
    
    def test_enqueue_dequeue_sequence(self):
        """Test complex enqueue-dequeue sequence."""
        self.queue.enqueue(1)
        self.queue.enqueue(2)
        self.assertEqual(self.queue.dequeue(), 1)
        self.queue.enqueue(3)
        self.queue.enqueue(4)
        self.assertEqual(self.queue.dequeue(), 2)
        self.assertEqual(self.queue.dequeue(), 3)
        self.assertEqual(self.queue.dequeue(), 4)
    
    def test_large_queue(self):
        """Test queue with large number of items."""
        large_queue = Queue()
        n = 10000
        for i in range(n):
            large_queue.enqueue(i)
        self.assertEqual(len(large_queue), n)
        self.assertEqual(large_queue.dequeue(), 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""
    
    def test_stack_with_none(self):
        """Test stack operations with None values."""
        stack = Stack()
        stack.push(None)
        self.assertIsNone(stack.peek())
        self.assertIsNone(stack.pop())
    
    def test_queue_with_none(self):
        """Test queue operations with None values."""
        queue = Queue()
        queue.enqueue(None)
        self.assertIsNone(queue.peek())
        self.assertIsNone(queue.dequeue())
    
    def test_stack_with_duplicates(self):
        """Test stack with duplicate values."""
        stack = Stack([1, 1, 1, 1])
        self.assertEqual(len(stack), 4)
        self.assertEqual(stack.pop(), 1)
        self.assertEqual(stack.pop(), 1)
    
    def test_queue_with_duplicates(self):
        """Test queue with duplicate values."""
        queue = Queue([1, 1, 1, 1])
        self.assertEqual(len(queue), 4)
        self.assertEqual(queue.dequeue(), 1)
        self.assertEqual(queue.dequeue(), 1)
    
    def test_stack_single_item_operations(self):
        """Test multiple operations on stack with single item."""
        stack = Stack([42])
        self.assertEqual(stack.peek(), 42)
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack.pop(), 42)
        self.assertTrue(stack.is_empty())
    
    def test_queue_single_item_operations(self):
        """Test multiple operations on queue with single item."""
        queue = Queue([42])
        self.assertEqual(queue.peek(), 42)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue.dequeue(), 42)
        self.assertTrue(queue.is_empty())


def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestStack))
    suite.addTests(loader.loadTestsFromTestCase(TestQueue))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    if result.wasSuccessful():
        print(" ALL TESTS PASSED!")
    else:
        print(" SOME TESTS FAILED!")
    
    return result


if __name__ == "__main__":
    run_tests()