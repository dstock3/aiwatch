import { Col, ListGroup } from 'react-bootstrap';

const BlogList = (props) => {

    return (
        <Col md={4}>
            <h4>Previous Blog Entries</h4>
            <ListGroup>
                {props.blogs.map((blog, index) => (
                    <ListGroup.Item key={index} action href={`#blog-entry-${index + 1}`}>
                        {blog.title}
                    </ListGroup.Item>
                ))}
            </ListGroup>
        </Col>
    );
}

export default BlogList;
