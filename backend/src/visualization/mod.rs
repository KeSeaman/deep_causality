//! Visualization module for causal graphs
//!
//! Exports causal graphs to Graphviz DOT format for visualization.

use std::io::Write;
use anyhow::Result;
use serde::Serialize;

/// Node in the causal graph
#[derive(Debug, Clone, Serialize)]
pub struct CausalNode {
    pub id: String,
    pub label: String,
    pub node_type: NodeType,
    pub score: Option<f64>,
}

/// Edge in the causal graph
#[derive(Debug, Clone, Serialize)]
pub struct CausalEdge {
    pub from: String,
    pub to: String,
    pub weight: f64,
    pub edge_type: EdgeType,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub enum NodeType {
    /// Feature/variable node
    Feature,
    /// Target/outcome node
    Target,
    /// Hidden/latent node
    Latent,
    /// Causal mechanism node
    Mechanism,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub enum EdgeType {
    /// Direct causal influence
    Causal,
    /// Redundant (shared) information
    Redundant,
    /// Synergistic (combined) influence
    Synergistic,
    /// Association (non-causal)
    Association,
}

/// A causal graph structure for visualization
#[derive(Debug, Clone, Serialize)]
pub struct CausalGraph {
    pub title: String,
    pub nodes: Vec<CausalNode>,
    pub edges: Vec<CausalEdge>,
}

impl CausalGraph {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_node(&mut self, id: impl Into<String>, label: impl Into<String>, node_type: NodeType) -> &mut Self {
        self.nodes.push(CausalNode {
            id: id.into(),
            label: label.into(),
            node_type,
            score: None,
        });
        self
    }

    pub fn add_node_with_score(&mut self, id: impl Into<String>, label: impl Into<String>, node_type: NodeType, score: f64) -> &mut Self {
        self.nodes.push(CausalNode {
            id: id.into(),
            label: label.into(),
            node_type,
            score: Some(score),
        });
        self
    }

    pub fn add_edge(&mut self, from: impl Into<String>, to: impl Into<String>, weight: f64, edge_type: EdgeType) -> &mut Self {
        self.edges.push(CausalEdge {
            from: from.into(),
            to: to.into(),
            weight,
            edge_type,
        });
        self
    }

    /// Build a graph from mRMR feature rankings
    pub fn from_mrmr_results(features: &[(String, f64)], target: &str) -> Self {
        let mut graph = Self::new(format!("mRMR Feature Selection → {}", target));
        
        // Add target node
        graph.add_node("target", target, NodeType::Target);
        
        // Add feature nodes with edges to target
        for (name, score) in features {
            let safe_id = name.replace(' ', "_").replace('-', "_").to_lowercase();
            graph.add_node_with_score(&safe_id, name, NodeType::Feature, *score);
            graph.add_edge(&safe_id, "target", *score, EdgeType::Causal);
        }
        
        graph
    }

    /// Export to DOT format (Graphviz)
    pub fn to_dot(&self) -> String {
        let mut dot = String::new();
        
        dot.push_str("digraph CausalGraph {\n");
        dot.push_str("  // Graph settings\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  bgcolor=\"#1a1a2e\";\n");
        dot.push_str("  fontcolor=\"white\";\n");
        dot.push_str(&format!("  label=\"{}\";\n", self.title));
        dot.push_str("  labelloc=\"t\";\n");
        dot.push_str("  fontname=\"Helvetica\";\n");
        dot.push_str("  fontsize=16;\n");
        dot.push_str("\n");
        
        dot.push_str("  // Default node style\n");
        dot.push_str("  node [\n");
        dot.push_str("    fontname=\"Helvetica\",\n");
        dot.push_str("    fontsize=10,\n");
        dot.push_str("    style=\"filled\",\n");
        dot.push_str("    fontcolor=\"white\"\n");
        dot.push_str("  ];\n\n");
        
        dot.push_str("  // Default edge style\n");
        dot.push_str("  edge [\n");
        dot.push_str("    fontname=\"Helvetica\",\n");
        dot.push_str("    fontsize=8,\n");
        dot.push_str("    color=\"#4a4a6a\"\n");
        dot.push_str("  ];\n\n");
        
        // Add nodes
        dot.push_str("  // Nodes\n");
        for node in &self.nodes {
            let (fillcolor, shape) = match node.node_type {
                NodeType::Target => ("#e94560", "oval"),
                NodeType::Feature => ("#0f3460", "box"),
                NodeType::Latent => ("#533483", "diamond"),
                NodeType::Mechanism => ("#16213e", "hexagon"),
            };
            
            let label = if let Some(score) = node.score {
                format!("{}\\n({:.3})", node.label, score)
            } else {
                node.label.clone()
            };
            
            dot.push_str(&format!(
                "  {} [label=\"{}\", fillcolor=\"{}\", shape={}];\n",
                node.id, label, fillcolor, shape
            ));
        }
        dot.push('\n');
        
        // Add edges
        dot.push_str("  // Edges\n");
        for edge in &self.edges {
            let color = match edge.edge_type {
                EdgeType::Causal => "#00ff88",
                EdgeType::Redundant => "#ff8800",
                EdgeType::Synergistic => "#00aaff",
                EdgeType::Association => "#888888",
            };
            
            let penwidth = 1.0 + edge.weight * 3.0;
            
            dot.push_str(&format!(
                "  {} -> {} [color=\"{}\", penwidth={:.1}, label=\"{:.2}\"];\n",
                edge.from, edge.to, color, penwidth, edge.weight
            ));
        }
        
        dot.push_str("}\n");
        dot
    }

    /// Write DOT file to disk
    pub fn write_dot(&self, path: &str) -> Result<()> {
        let mut file = std::fs::File::create(path)?;
        file.write_all(self.to_dot().as_bytes())?;
        Ok(())
    }

    /// Export to JSON for web visualization
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(&self)?)
    }
}

/// Graphviz exporter utility
pub struct GraphvizExporter;

impl GraphvizExporter {
    /// Generate SVG from DOT (requires graphviz installed)
    pub fn dot_to_svg(dot_path: &str, svg_path: &str) -> Result<()> {
        use std::process::Command;
        
        let output = Command::new("dot")
            .args(["-Tsvg", dot_path, "-o", svg_path])
            .output()?;
        
        if !output.status.success() {
            anyhow::bail!("Graphviz failed: {}", String::from_utf8_lossy(&output.stderr));
        }
        
        Ok(())
    }

    /// Generate PNG from DOT
    pub fn dot_to_png(dot_path: &str, png_path: &str) -> Result<()> {
        use std::process::Command;
        
        let output = Command::new("dot")
            .args(["-Tpng", dot_path, "-o", png_path])
            .output()?;
        
        if !output.status.success() {
            anyhow::bail!("Graphviz failed: {}", String::from_utf8_lossy(&output.stderr));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_from_mrmr() {
        let features = vec![
            ("ICULOS".to_string(), 1.0),
            ("HR".to_string(), 0.8),
            ("MAP".to_string(), 0.6),
        ];
        
        let graph = CausalGraph::from_mrmr_results(&features, "SepsisLabel");
        assert_eq!(graph.nodes.len(), 4); // 3 features + 1 target
        assert_eq!(graph.edges.len(), 3);
        
        let dot = graph.to_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains("ICULOS"));
    }

    #[test]
    fn test_dot_format() {
        let mut graph = CausalGraph::new("Test Graph");
        graph.add_node("a", "Feature A", NodeType::Feature);
        graph.add_node("b", "Target", NodeType::Target);
        graph.add_edge("a", "b", 0.5, EdgeType::Causal);
        
        let dot = graph.to_dot();
        assert!(dot.contains("a -> b"));
    }
}
