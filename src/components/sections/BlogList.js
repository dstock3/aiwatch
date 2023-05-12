import React from 'react';
import { Col, ListGroup } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import '../../style/blog.css';

const BlogList = ({ blogs }) => {
  const [activeIndex, setActiveIndex] = React.useState(null);

  const handleMouseEnter = (index) => {
    setActiveIndex(index);
  };

  const handleMouseLeave = () => {
    setActiveIndex(null);
  };

  return (
    <Col md={3} className="blog-list">
      <h4 className="text-center mb-4">Previous Entries</h4>
      <ListGroup>
        {blogs.map((blog, index) => (
          <Link to={`/blog/${blog.id}`} key={index}>
            <ListGroup.Item
              onMouseEnter={() => handleMouseEnter(index)}
              onMouseLeave={() => handleMouseLeave()}
              active={activeIndex === index}
            >
              <div className="bloglist-item d-flex justify-content-between align-items-center">
                <div className="title">{blog.title}</div>
                <div className="date">{blog.date}</div>
              </div>
            </ListGroup.Item>
          </Link>
        ))}
      </ListGroup>
    </Col>
  );
};

export default BlogList;
