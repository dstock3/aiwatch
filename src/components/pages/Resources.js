import React, { useState } from 'react';
import { Container, Row, Col, Card, Form, FormControl, Button } from 'react-bootstrap';
import { Helmet } from 'react-helmet';
import '../../style/resources.css';
import resources from '../../data/resources';

const Resources = () => {
  const [searchKeyword, setSearchKeyword] = useState('');

  const handleKeywordClick = (keyword) => {
    setSearchKeyword(keyword);
  };

  const filteredResources = resources.filter((resource) => {
    if (searchKeyword === '') return true;
  
    const searchWords = searchKeyword.toLowerCase().split(' ');
    return searchWords.some((word) =>
      resource.keywords.some((keyword) => keyword.toLowerCase().includes(word))
    );
  });
  
  const handleSearchChange = (event) => {
    setSearchKeyword(event.target.value);
  };

  return (
    <>
      <Helmet>
        <title>Resources - AI Watch</title>
      </Helmet>
      <Container className="my-5 resources-container">
        <Row className="resources-head align-items-center mb-4 gradient text-white">
          <Col xs={12} md={6}>
            <h1>Resources</h1>
          </Col>
          <Col xs={12} md={6}>
            <Form className="d-flex search">
              <FormControl
                type="search"
                placeholder="Search by keyword"
                className="mr-2 search-input"
                value={searchKeyword}
                onChange={handleSearchChange}
              />
              <Button variant="outline-primary" className="search-button">
                Search
              </Button>
            </Form>
          </Col>
        </Row>
        <Row>
          {filteredResources.map((resource, index) => (
            <Col key={index} xs={12} md={4} className="mb-4">
              <Card className="resource-container gradient text-white">
                <Card.Body className="resource-card-body">
                  <Card.Title>
                    <a href={resource.url} target="_blank" rel="noopener noreferrer">{resource.title}</a>
                  </Card.Title>
                  <hr className="horizontal-rule" />
                  <Card.Text>{resource.description}</Card.Text>

                  {resource.keywords && (
                    <div className="mt-2">
                      {resource.keywords.map((keyword, idx) => (
                        <span
                          key={idx}
                          className="keyword-box"
                          onClick={() => handleKeywordClick(keyword)}
                        >
                          {keyword}
                        </span>
                      ))}
                    </div>
                  )}
                </Card.Body>
              </Card>
            </Col>
          ))}
        </Row>
      </Container>
    </>
  );
};

export default Resources;