
#Part 2: Manim Decision Tree Animation

from manim import *
import numpy as np

class GiniImpurityAnimation(Scene):
    """Visualize Gini Impurity concept"""
    
    def construct(self):
        # Title
        title = Text("Gini Impurity", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Formula
        formula = MathTex(
            r"Gini = 1 - \sum_{i=1}^{n} p_i^2",
            font_size=40
        )
        formula.next_to(title, DOWN, buff=0.5)
        self.play(Write(formula))
        self.wait()
        
        # Show three scenarios
        scenarios = VGroup()
        
        # Pure node
        pure_circles = VGroup(*[
            Circle(radius=0.2, color=BLUE, fill_opacity=0.8)
            for _ in range(5)
        ]).arrange(RIGHT, buff=0.2)
        pure_label = Text("Pure: Gini = 0", font_size=24)
        pure_calc = MathTex(r"1 - (1^2 + 0^2) = 0", font_size=20)
        pure_group = VGroup(pure_circles, pure_label, pure_calc).arrange(DOWN, buff=0.3)
        
        # Mixed node
        mixed_circles = VGroup(
            *[Circle(radius=0.2, color=BLUE, fill_opacity=0.8) for _ in range(3)],
            *[Circle(radius=0.2, color=RED, fill_opacity=0.8) for _ in range(2)]
        ).arrange(RIGHT, buff=0.2)
        mixed_label = Text("Mixed: Gini = 0.48", font_size=24)
        mixed_calc = MathTex(r"1 - (0.6^2 + 0.4^2) = 0.48", font_size=20)
        mixed_group = VGroup(mixed_circles, mixed_label, mixed_calc).arrange(DOWN, buff=0.3)
        
        # Most impure
        impure_circles = VGroup(
            *[Circle(radius=0.2, color=BLUE, fill_opacity=0.8) for _ in range(3)],
            *[Circle(radius=0.2, color=RED, fill_opacity=0.8) for _ in range(3)]
        ).arrange(RIGHT, buff=0.2)
        impure_label = Text("Impure: Gini = 0.5", font_size=24)
        impure_calc = MathTex(r"1 - (0.5^2 + 0.5^2) = 0.5", font_size=20)
        impure_group = VGroup(impure_circles, impure_label, impure_calc).arrange(DOWN, buff=0.3)
        
        scenarios = VGroup(pure_group, mixed_group, impure_group)
        scenarios.arrange(DOWN, buff=0.8)
        scenarios.next_to(formula, DOWN, buff=0.8)
        scenarios.scale(0.8)
        
        for group in scenarios:
            self.play(FadeIn(group), run_time=1.5)
            self.wait(0.5)
        
        self.wait(2)


class TreeGrowingAnimation(Scene):
    """Animate how a decision tree grows"""
    
    def construct(self):
        # Title
        title = Text("Decision Tree Growing Process", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Root node
        root = self.create_node("Root\nGini = 0.67", YELLOW)
        root.move_to(UP * 2)
        self.play(FadeIn(root))
        self.wait()
        
        # Show split question
        question = Text("Petal Length ≤ 2.5?", font_size=24, color=GREEN)
        question.next_to(root, DOWN, buff=0.3)
        self.play(Write(question))
        self.wait()
        
        # Create child nodes
        left_child = self.create_node("Setosa\nGini = 0.0", BLUE)
        right_child = self.create_node("Mixed\nGini = 0.5", ORANGE)
        
        left_child.next_to(root, DOWN + LEFT * 2, buff=1)
        right_child.next_to(root, DOWN + RIGHT * 2, buff=1)
        
        # Create edges
        left_edge = Line(root.get_bottom(), left_child.get_top(), color=WHITE)
        right_edge = Line(root.get_bottom(), right_child.get_top(), color=WHITE)
        
        left_label = Text("Yes", font_size=20, color=GREEN).next_to(left_edge, LEFT, buff=0.1)
        right_label = Text("No", font_size=20, color=RED).next_to(right_edge, RIGHT, buff=0.1)
        
        self.play(
            Create(left_edge),
            Create(right_edge),
            FadeOut(question)
        )
        self.play(
            FadeIn(left_child),
            FadeIn(right_child),
            Write(left_label),
            Write(right_label)
        )
        self.wait(2)
        
        # Split right child further
        question2 = Text("Petal Width ≤ 1.7?", font_size=20, color=GREEN)
        question2.next_to(right_child, DOWN, buff=0.2)
        self.play(Write(question2))
        self.wait()
        
        # Create grandchildren
        versicolor = self.create_node("Versicolor\nGini = 0.17", GREEN, scale=0.8)
        virginica = self.create_node("Virginica\nGini = 0.04", PURPLE, scale=0.8)
        
        versicolor.next_to(right_child, DOWN + LEFT, buff=1)
        virginica.next_to(right_child, DOWN + RIGHT, buff=1)
        
        edge3 = Line(right_child.get_bottom(), versicolor.get_top(), color=WHITE)
        edge4 = Line(right_child.get_bottom(), virginica.get_top(), color=WHITE)
        
        self.play(
            Create(edge3),
            Create(edge4),
            FadeOut(question2)
        )
        self.play(
            FadeIn(versicolor),
            FadeIn(virginica)
        )
        self.wait(3)
    
    def create_node(self, text, color, scale=1.0):
        """Helper to create a tree node"""
        rect = RoundedRectangle(
            width=2, height=1, corner_radius=0.2,
            color=color, fill_opacity=0.3
        )
        label = Text(text, font_size=20)
        node = VGroup(rect, label)
        node.scale(scale)
        return node


class PredictionPathAnimation(Scene):
    """Animate how a prediction is made"""
    
    def construct(self):
        title = Text("Making a Prediction", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Sample data point
        sample_text = Text(
            "Sample: Petal Length = 5.0, Petal Width = 1.5",
            font_size=24,
            color=YELLOW
        )
        sample_text.next_to(title, DOWN, buff=0.5)
        self.play(Write(sample_text))
        self.wait()
        
        # Build simple tree
        root = self.create_decision_node("Petal Length ≤ 2.5?", YELLOW)
        root.move_to(UP * 1)
        
        left = self.create_leaf_node("Setosa", BLUE)
        left.next_to(root, DOWN + LEFT * 2, buff=1)
        
        right = self.create_decision_node("Petal Width ≤ 1.7?", ORANGE)
        right.next_to(root, DOWN + RIGHT * 2, buff=1)
        
        versicolor = self.create_leaf_node("Versicolor", GREEN)
        versicolor.next_to(right, DOWN + LEFT, buff=1)
        
        virginica = self.create_leaf_node("Virginica", PURPLE)
        virginica.next_to(right, DOWN + RIGHT, buff=1)
        
        # Create edges
        edges = [
            Line(root.get_bottom(), left.get_top()),
            Line(root.get_bottom(), right.get_top()),
            Line(right.get_bottom(), versicolor.get_top()),
            Line(right.get_bottom(), virginica.get_top())
        ]
        
        # Draw tree
        tree_group = VGroup(root, left, right, versicolor, virginica, *edges)
        tree_group.scale(0.7).shift(DOWN * 0.5)
        
        self.play(FadeIn(tree_group))
        self.wait()
        
        # Animate path
        path_nodes = [root, right, versicolor]
        path_edges = [edges[1], edges[2]]
        
        for i, node in enumerate(path_nodes):
            # Highlight node
            highlight = node.copy()
            highlight.set_stroke(YELLOW, width=5)
            self.play(Create(highlight), run_time=0.5)
            self.wait(0.5)
            
            # Highlight edge to next node
            if i < len(path_edges):
                edge_highlight = path_edges[i].copy()
                edge_highlight.set_stroke(YELLOW, width=5)
                self.play(Create(edge_highlight), run_time=0.5)
        
        # Show result
        result = Text("Prediction: Versicolor", font_size=32, color=GREEN)
        result.to_edge(DOWN, buff=1)
        self.play(Write(result))
        self.wait(3)
    
    def create_decision_node(self, text, color):
        rect = RoundedRectangle(
            width=2.5, height=0.8, corner_radius=0.2,
            color=color, fill_opacity=0.2
        )
        label = Text(text, font_size=18)
        return VGroup(rect, label)
    
    def create_leaf_node(self, text, color):
        circle = Circle(radius=0.6, color=color, fill_opacity=0.3)
        label = Text(text, font_size=20)
        return VGroup(circle, label)


class InformationGainAnimation(Scene):
    """Visualize Information Gain calculation"""
    
    def construct(self):
        title = Text("Information Gain", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Formula
        formula = MathTex(
            r"IG = H(parent) - \sum \frac{n_{child}}{n_{parent}} H(child)",
            font_size=36
        )
        formula.next_to(title, DOWN, buff=0.5)
        self.play(Write(formula))
        self.wait()
        
        # Visual representation
        # Parent node
        parent_rect = Rectangle(width=6, height=1, color=YELLOW, fill_opacity=0.3)
        parent_label = Text("Parent: 100 samples\nGini = 0.67", font_size=20)
        parent = VGroup(parent_rect, parent_label)
        parent.move_to(UP * 1.5)
        
        # Draw samples in parent
        blue_dots = [Dot(color=BLUE).scale(0.5) for _ in range(33)]
        red_dots = [Dot(color=RED).scale(0.5) for _ in range(33)]
        green_dots = [Dot(color=GREEN).scale(0.5) for _ in range(34)]
        
        all_dots = VGroup(*blue_dots, *red_dots, *green_dots)
        all_dots.arrange_in_grid(rows=5, buff=0.1)
        all_dots.move_to(parent_rect.get_center())
        
        self.play(FadeIn(parent))
        self.play(FadeIn(all_dots))
        self.wait()
        
        # Split into children
        left_rect = Rectangle(width=2.5, height=1, color=BLUE, fill_opacity=0.3)
        left_label = Text("Left: 33\nGini = 0.0", font_size=18)
        left_child = VGroup(left_rect, left_label)
        left_child.next_to(parent, DOWN + LEFT * 2, buff=1)
        
        right_rect = Rectangle(width=3.5, height=1, color=ORANGE, fill_opacity=0.3)
        right_label = Text("Right: 67\nGini = 0.5", font_size=18)
        right_child = VGroup(right_rect, right_label)
        right_child.next_to(parent, DOWN + RIGHT * 1.5, buff=1)
        
        # Animate split
        left_dots = VGroup(*blue_dots)
        right_dots = VGroup(*red_dots, *green_dots)
        
        self.play(
            left_dots.animate.move_to(left_rect.get_center()).scale(0.7),
            right_dots.animate.move_to(right_rect.get_center()).scale(0.7),
            FadeIn(left_child),
            FadeIn(right_child)
        )
        self.wait()
        
        # Calculate IG
        calc = MathTex(
            r"IG &= 0.67 - \left(\frac{33}{100} \times 0.0 + \frac{67}{100} \times 0.5\right)\\",
            r"&= 0.67 - 0.335\\",
            r"&= 0.335",
            font_size=30
        )
        calc.to_edge(DOWN, buff=0.5)
        
        self.play(Write(calc))
        self.wait(3)


# Main scene combining all animations
class DecisionTreeAnimation(Scene):
    """Complete decision tree educational video"""
    
    def construct(self):
        # Play all sub-scenes
        scenes = [
            GiniImpurityAnimation,
            InformationGainAnimation,
            TreeGrowingAnimation,
            PredictionPathAnimation
        ]
        
        for scene_class in scenes:
            scene = scene_class()
            scene.construct()
            self.wait(2)
            self.clear()


# USAGE:
# Save this file as decision_tree_manim.py
# Run: manim -pql decision_tree_manim.py GiniImpurityAnimation
# Or: manim -pql decision_tree_manim.py TreeGrowingAnimation
# Or: manim -pqh decision_tree_manim.py DecisionTreeAnimation  # High quality, all scenes
