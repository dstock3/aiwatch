import React from 'react';
import { Container } from 'react-bootstrap';
import '../../App.css'

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="py-3">
      <Container>
        <p className="text-center mb-0">
          Copyright &copy; {currentYear} AI Watch
        </p>
      </Container>
    </footer>
  );
};

export default Footer;
