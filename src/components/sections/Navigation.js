import React from 'react';
import { Navbar, Nav, Container } from 'react-bootstrap';
import { LinkContainer } from 'react-router-bootstrap';

const Navigation = () => {
  return (
    <>
      <Navbar collapseOnSelect expand="lg">
        <Container>
          <LinkContainer to="/">
            <Navbar.Brand id="ai-watch-brand">AI Watch</Navbar.Brand>
          </LinkContainer>
          <Navbar.Toggle aria-controls="responsive-navbar-nav" />
          <Navbar.Collapse id="responsive-navbar-nav">
            <Nav className="me-auto">
              <LinkContainer to="/videos">
                <Nav.Link className="ai-watch-link">Videos</Nav.Link>
              </LinkContainer>
              <LinkContainer to="/blog">
                <Nav.Link className="ai-watch-link">Blog</Nav.Link>
              </LinkContainer>
              <LinkContainer to="/resources">
                <Nav.Link className="ai-watch-link">Resources</Nav.Link>
              </LinkContainer>
              <LinkContainer to="/contact">
                <Nav.Link className="ai-watch-link">Contact</Nav.Link>
              </LinkContainer>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>
      <hr className="rule"></hr>
    </>
  );
};

export default Navigation;