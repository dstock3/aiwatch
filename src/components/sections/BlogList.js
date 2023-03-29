import React from 'react';
import { Col, ListGroup } from 'react-bootstrap';
import '../../style/blog.css'

const BlogList = (props) => {
    const [activeIndex, setActiveIndex] = React.useState(null);

    const handleMouseEnter = (index) => {
        setActiveIndex(index);
    };

    const handleMouseLeave = () => {
        setActiveIndex(null);
    };

    return (
        <Col md={3}>
            <h4 className="text-center mb-4">Previous Blog Entries</h4>
            <ListGroup>
                {props.blogs.map((blog, index) => (
                    <ListGroup.Item
                        key={index}
                        onClick={() => props.onBlogSelect(blog)}
                        onMouseEnter={() => handleMouseEnter(index)}
                        onMouseLeave={() => handleMouseLeave()}
                        active={activeIndex === index}
                    >
                        <div className="bloglist-item d-flex justify-content-between align-items-center">
                            <div className="title">
                                {blog.title}
                            </div>
                            <div className="date">
                                {blog.date}
                            </div>
                        </div>
                    </ListGroup.Item>
                ))}
            </ListGroup>
        </Col>
    );
}

export default BlogList;